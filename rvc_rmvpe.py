from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import subprocess
import tempfile

from audio_transposer import (
    AudioTransposeError,
    find_ffmpeg,
    rubberband_filter_available,
    transpose_audio,
)
from seed_vc_svc import (
    _encode_single_vocal,
    _mix_stems,
    ensure_stem_pair,
)
from vocal_separator import DEFAULT_MODEL


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]

RVC_REPOSITORY = (
    "https://github.com/"
    "RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
)
RVC_PINNED_COMMIT = (
    "81eed5e8f68b6bed1789f682fe78cdd324495afc"
)


class RVCRMVPEError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent


def rvc_repo_dir() -> Path:
    return project_root() / "tools" / "rvc"


def rvc_venv_dir() -> Path:
    return project_root() / ".venv_rvc"


def rvc_python() -> Path:
    if os.name == "nt":
        return (
            rvc_venv_dir()
            / "Scripts"
            / "python.exe"
        )
    return (
        rvc_venv_dir()
        / "bin"
        / "python"
    )


def rvc_cli_path() -> Path:
    return (
        rvc_repo_dir()
        / "infer"
        / "cli.py"
    )


def rvc_hubert_path() -> Path:
    return (
        rvc_repo_dir()
        / "assets"
        / "hubert_base"
        / "pytorch_model.bin"
    )


def rvc_rmvpe_path() -> Path:
    return (
        rvc_repo_dir()
        / "assets"
        / "rmvpe"
        / "rmvpe.pt"
    )


def rvc_log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_rmvpe_last.log"
    )


def rvc_available() -> bool:
    return all(
        (
            rvc_python().is_file(),
            rvc_cli_path().is_file(),
            rvc_hubert_path().is_file(),
            rvc_rmvpe_path().is_file(),
        )
    )


def rvc_status_text() -> str:
    missing: list[str] = []

    if not rvc_python().is_file():
        missing.append(".venv_rvc")

    if not rvc_cli_path().is_file():
        missing.append("tools\\rvc")

    if not rvc_hubert_path().is_file():
        missing.append("HuBERT")

    if not rvc_rmvpe_path().is_file():
        missing.append("RMVPE")

    if missing:
        return (
            "RVC + RMVPE 미설치/불완전: "
            + ", ".join(missing)
            + "\nSETUP_RVC_RMVPE_GPU.bat을 실행하세요."
        )

    return (
        "RVC + RMVPE 사용 가능 "
        f"(pinned {RVC_PINNED_COMMIT[:8]})"
    )


def _creationflags() -> int:
    return getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )


def _emit(
    callback: LogCallback | None,
    message: str,
) -> None:
    clean = str(message).strip()

    if clean and callback:
        callback(clean)


def _progress(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback:
        callback(
            max(
                0,
                min(
                    100,
                    int(percent),
                ),
            ),
            str(message),
        )


def _rvc_env() -> dict[str, str]:
    env = os.environ.copy()

    env.setdefault(
        "PYTHONUTF8",
        "1",
    )
    env.setdefault(
        "PYTHONIOENCODING",
        "utf-8",
    )

    ffmpeg = find_ffmpeg()

    if ffmpeg:
        ffmpeg_dir = str(
            Path(ffmpeg).resolve().parent
        )
        current_path = env.get(
            "PATH",
            "",
        )

        if ffmpeg_dir.lower() not in (
            current_path.lower()
        ):
            env["PATH"] = (
                ffmpeg_dir
                + os.pathsep
                + current_path
            )

    return env


def _effective_index(
    index_path: str | Path | None,
    index_rate: float,
    log_callback: LogCallback | None,
) -> tuple[Path | None, float]:
    rate = max(
        0.0,
        min(
            1.0,
            float(index_rate),
        ),
    )

    if rate <= 0:
        return None, 0.0

    if not index_path:
        _emit(
            log_callback,
            (
                "RVC index 미선택: "
                "안전하게 index rate를 0으로 사용합니다."
            ),
        )
        return None, 0.0

    index = Path(
        index_path
    ).expanduser().resolve()

    if not index.is_file():
        _emit(
            log_callback,
            (
                "선택한 RVC index를 찾을 수 없어 "
                "index rate를 0으로 사용합니다: "
                f"{index}"
            ),
        )
        return None, 0.0

    return index, rate


def run_rvc_vocal(
    input_vocal: str | Path,
    output_wav: str | Path,
    *,
    model_path: str | Path,
    index_path: str | Path | None = None,
    semitones: int = 0,
    index_rate: float = 0.75,
    protect: float = 0.33,
    rms_mix_rate: float = 1.0,
    speaker_id: int = 0,
    log_callback: LogCallback | None = None,
) -> Path:
    if not rvc_available():
        raise RVCRMVPEError(
            rvc_status_text()
        )

    input_vocal = Path(
        input_vocal
    ).resolve()
    output_wav = Path(
        output_wav
    ).resolve()
    model = Path(
        model_path
    ).expanduser().resolve()

    if not input_vocal.is_file():
        raise FileNotFoundError(
            input_vocal
        )

    if (
        not model.is_file()
        or model.suffix.lower()
        != ".pth"
    ):
        raise RVCRMVPEError(
            "RVC 모델 .pth 파일을 찾을 수 없습니다.\n"
            f"{model}"
        )

    output_wav.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index, effective_rate = (
        _effective_index(
            index_path,
            index_rate,
            log_callback,
        )
    )

    protect = max(
        0.0,
        min(
            0.5,
            float(protect),
        ),
    )
    rms_mix_rate = max(
        0.0,
        min(
            1.0,
            float(rms_mix_rate),
        ),
    )

    command = [
        str(
            rvc_python()
        ),
        str(
            rvc_cli_path()
        ),
        "--model",
        str(model),
        "--input",
        str(input_vocal),
        "--output",
        str(output_wav),
        "--pitch",
        str(int(semitones)),
        "--f0-method",
        "rmvpe",
        "--index-rate",
        f"{effective_rate:.4f}",
        "--rms-mix-rate",
        f"{rms_mix_rate:.4f}",
        "--protect",
        f"{protect:.4f}",
        "--speaker-id",
        str(
            max(
                0,
                int(speaker_id),
            )
        ),
        "--format",
        "wav",
        "--overwrite",
    ]

    if index is not None:
        command.extend(
            [
                "--index",
                str(index),
            ]
        )

    _emit(
        log_callback,
        (
            "RVC + RMVPE 추론 시작: "
            f"pitch={int(semitones):+d}, "
            f"index_rate={effective_rate:.2f}, "
            f"protect={protect:.2f}, "
            f"speaker_id={int(speaker_id)}"
        ),
    )
    _emit(
        log_callback,
        f"RVC 모델: {model}",
    )

    if index is not None:
        _emit(
            log_callback,
            f"RVC index: {index}",
        )
    else:
        _emit(
            log_callback,
            "RVC index: 미사용",
        )

    process = subprocess.Popen(
        command,
        cwd=str(
            rvc_repo_dir()
        ),
        env=_rvc_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_creationflags(),
    )

    lines: list[str] = []

    assert process.stdout is not None

    for raw_line in process.stdout:
        for part in raw_line.replace(
            "\r",
            "\n",
        ).splitlines():
            clean = part.strip()

            if not clean:
                continue

            lines.append(clean)
            _emit(
                log_callback,
                clean,
            )

    return_code = process.wait()

    log_file = rvc_log_path()
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        log_file.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    if (
        return_code != 0
        or not output_wav.is_file()
        or output_wav.stat().st_size <= 0
    ):
        raise RVCRMVPEError(
            "RVC + RMVPE 추론에 실패했습니다.\n\n"
            f"로그: {log_file}\n\n"
            + "\n".join(
                lines[-100:]
            )[-8000:]
        )

    _emit(
        log_callback,
        f"RVC 보컬 생성 완료: {output_wav}",
    )

    return output_wav


def convert_vocal_rvc(
    vocal_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    index_path: str | Path | None = None,
    semitones: int = 0,
    index_rate: float = 0.75,
    protect: float = 0.33,
    rms_mix_rate: float = 1.0,
    speaker_id: int = 0,
    progress: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> Path:
    vocal = Path(
        vocal_path
    ).resolve()
    output = Path(
        output_path
    ).resolve()

    if not vocal.is_file():
        raise FileNotFoundError(
            vocal
        )

    _progress(
        progress,
        5,
        "RVC + RMVPE 보컬 변환 준비 중...",
    )

    with tempfile.TemporaryDirectory(
        prefix="vocal_pitch_rvc_"
    ) as temp_name:
        temp_dir = Path(
            temp_name
        )

        converted_wav = (
            temp_dir
            / "rvc_vocal.wav"
        )

        _progress(
            progress,
            15,
            "RMVPE F0 추출 + RVC 음색 변환 중...",
        )

        run_rvc_vocal(
            vocal,
            converted_wav,
            model_path=model_path,
            index_path=index_path,
            semitones=semitones,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            log_callback=log_callback,
        )

        _progress(
            progress,
            90,
            "RVC 보컬 출력 인코딩 중...",
        )

        _encode_single_vocal(
            converted_wav,
            output,
        )

    _progress(
        progress,
        100,
        f"RVC 보컬 변환 완료: {output.name}",
    )

    return output


def convert_full_mix_rvc(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    index_path: str | Path | None = None,
    semitones: int = 0,
    index_rate: float = 0.75,
    protect: float = 0.33,
    rms_mix_rate: float = 1.0,
    speaker_id: int = 0,
    separator_model: str = DEFAULT_MODEL,
    separator_cache: bool = True,
    separator_autocast: bool = True,
    progress: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> Path:
    input_file = Path(
        input_path
    ).resolve()
    output = Path(
        output_path
    ).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(
            input_file
        )

    if not rvc_available():
        raise RVCRMVPEError(
            rvc_status_text()
        )

    rb_ok, rb_status = (
        rubberband_filter_available()
    )

    if not rb_ok:
        raise RVCRMVPEError(
            "원곡 전체 RVC 변환은 반주도 같은 키로 이동해야 합니다.\n"
            + rb_status
        )

    _progress(
        progress,
        3,
        "RVC용 보컬/반주 stem 준비 중...",
    )

    pair = ensure_stem_pair(
        input_file,
        model_filename=separator_model,
        use_cache=separator_cache,
        use_autocast=separator_autocast,
        log_callback=log_callback,
    )

    _progress(
        progress,
        25,
        (
            "보컬/반주 캐시 준비 완료."
            if pair.cache_hit
            else "보컬/반주 분리 완료."
        ),
    )

    with tempfile.TemporaryDirectory(
        prefix="vocal_pitch_rvc_mix_"
    ) as temp_name:
        temp_dir = Path(
            temp_name
        )

        converted_vocal = (
            temp_dir
            / "rvc_vocal.wav"
        )

        _progress(
            progress,
            30,
            "RMVPE F0 추출 + RVC 남성/타깃 음색 변환 중...",
        )

        run_rvc_vocal(
            pair.vocals_path,
            converted_vocal,
            model_path=model_path,
            index_path=index_path,
            semitones=semitones,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            log_callback=log_callback,
        )

        _progress(
            progress,
            78,
            "반주를 같은 키로 이동하는 중...",
        )

        shifted_instrumental = (
            temp_dir
            / "instrumental_shifted.wav"
        )

        def instrumental_progress(
            percent: int,
            text: str,
        ) -> None:
            mapped = (
                78
                + int(
                    max(
                        0,
                        min(
                            100,
                            percent,
                        ),
                    )
                    * 0.12
                )
            )

            _progress(
                progress,
                min(
                    90,
                    mapped,
                ),
                "반주 키 변경: " + text,
            )

        try:
            transpose_audio(
                pair.instrumental_path,
                shifted_instrumental,
                semitones=semitones,
                preserve_formant=False,
                quality="quality",
                progress=instrumental_progress,
            )
        except AudioTransposeError as exc:
            raise RVCRMVPEError(
                "RVC 보컬은 생성됐지만 반주 키 변경에 실패했습니다.\n\n"
                f"{exc}"
            ) from exc

        _progress(
            progress,
            92,
            "RVC 보컬과 변환 반주를 재합성하는 중...",
        )

        _mix_stems(
            converted_vocal,
            shifted_instrumental,
            pair.vocals_path,
            output,
            log_callback=log_callback,
        )

    _progress(
        progress,
        100,
        f"RVC + RMVPE 전체곡 변환 완료: {output.name}",
    )

    return output
