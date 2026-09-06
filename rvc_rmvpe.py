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
from rvc_harmony_guard import blend_adaptive_vocals

# V24_ADAPTIVE_RVC_HARMONY_GUARD_PATCH


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

    rvc_root = str(
        rvc_repo_dir().resolve()
    )
    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    pythonpath_entries = [
        rvc_root,
    ]

    if existing_pythonpath:
        pythonpath_entries.append(
            existing_pythonpath
        )

    env["PYTHONPATH"] = os.pathsep.join(
        pythonpath_entries
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

    # RVC_INFERENCE_MODULE_LAUNCH_HOTFIX
    # Run the CLI as a module so `infer` resolves as the package.
    command = [
        str(
            rvc_python()
        ),
        "-m",
        "infer.cli",
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


def _prepare_adaptive_rvc_vocal(
    source_vocal: Path,
    temp_dir: Path,
    *,
    model_path: str | Path,
    index_path: str | Path | None,
    semitones: int,
    index_rate: float,
    protect: float,
    rms_mix_rate: float,
    speaker_id: int,
    harmony_guard_enabled: bool,
    harmony_guard_sensitivity: str,
    harmony_guard_crossfade_ms: int,
    progress: ProgressCallback | None,
    progress_points: tuple[int, int, int, int],
    log_callback: LogCallback | None,
) -> Path:
    start_percent, rvc_percent, fallback_percent, blend_percent = (
        progress_points
    )

    raw_rvc = (
        temp_dir
        / "rvc_vocal_raw.wav"
    )

    _progress(
        progress,
        start_percent,
        "RMVPE F0 추출 + RVC 음색 변환 중...",
    )

    run_rvc_vocal(
        source_vocal,
        raw_rvc,
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
        rvc_percent,
        "RVC 보컬 생성 완료.",
    )

    if not harmony_guard_enabled:
        _emit(
            log_callback,
            "[Harmony Guard] 비활성화 - 기존 RVC 출력 그대로 사용",
        )
        return raw_rvc

    rb_ok, rb_status = (
        rubberband_filter_available()
    )

    if not rb_ok:
        _emit(
            log_callback,
            (
                "[Harmony Guard] Pitch-only 우회용 RubberBand를 사용할 수 없어 "
                "Harmony Guard를 건너뜁니다: "
                + rb_status
            ),
        )
        return raw_rvc

    pitch_only = (
        temp_dir
        / "pitch_only_vocal.wav"
    )

    _progress(
        progress,
        min(
            fallback_percent,
            99,
        ),
        "Harmony Guard: 안전 우회용 Pitch-only 보컬 생성 중...",
    )

    try:
        transpose_audio(
            source_vocal,
            pitch_only,
            semitones=semitones,
            preserve_formant=True,
            quality="quality",
        )
    except AudioTransposeError as exc:
        _emit(
            log_callback,
            (
                "[Harmony Guard] Pitch-only 보컬 생성 실패 - "
                "기존 RVC 출력으로 계속합니다: "
                f"{exc}"
            ),
        )
        return raw_rvc

    adaptive = (
        temp_dir
        / "rvc_vocal_harmony_guard.wav"
    )

    _progress(
        progress,
        min(
            blend_percent,
            99,
        ),
        "Harmony Guard: 화음/코러스 위험도 분석 + Adaptive Blend 중...",
    )

    try:
        report = blend_adaptive_vocals(
            source_vocal,
            raw_rvc,
            pitch_only,
            adaptive,
            sensitivity=harmony_guard_sensitivity,
            crossfade_ms=harmony_guard_crossfade_ms,
            log_callback=log_callback,
        )
    except Exception as exc:
        _emit(
            log_callback,
            (
                "[Harmony Guard] 분석/Blend 실패 - "
                "기존 RVC 출력으로 계속합니다: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        return raw_rvc

    _emit(
        log_callback,
        (
            "[Harmony Guard] 적용 완료: "
            f"위험 구간 {report.risky_region_count}개, "
            f"Pitch-only 우회 {report.fallback_seconds:.1f}s, "
            f"부분 Blend {report.blend_seconds:.1f}s"
        ),
    )

    return adaptive


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
    harmony_guard_enabled: bool = True,
    harmony_guard_sensitivity: str = "medium",
    harmony_guard_crossfade_ms: int = 500,
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
            _prepare_adaptive_rvc_vocal(
                vocal,
                temp_dir,
                model_path=model_path,
                index_path=index_path,
                semitones=semitones,
                index_rate=index_rate,
                protect=protect,
                rms_mix_rate=rms_mix_rate,
                speaker_id=speaker_id,
                harmony_guard_enabled=bool(
                    harmony_guard_enabled
                ),
                harmony_guard_sensitivity=str(
                    harmony_guard_sensitivity
                ),
                harmony_guard_crossfade_ms=int(
                    harmony_guard_crossfade_ms
                ),
                progress=progress,
                progress_points=(
                    15,
                    68,
                    76,
                    86,
                ),
                log_callback=log_callback,
            )
        )

        _progress(
            progress,
            92,
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
    harmony_guard_enabled: bool = True,
    harmony_guard_sensitivity: str = "medium",
    harmony_guard_crossfade_ms: int = 500,
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
            _prepare_adaptive_rvc_vocal(
                Path(
                    pair.vocals_path
                ),
                temp_dir,
                model_path=model_path,
                index_path=index_path,
                semitones=semitones,
                index_rate=index_rate,
                protect=protect,
                rms_mix_rate=rms_mix_rate,
                speaker_id=speaker_id,
                harmony_guard_enabled=bool(
                    harmony_guard_enabled
                ),
                harmony_guard_sensitivity=str(
                    harmony_guard_sensitivity
                ),
                harmony_guard_crossfade_ms=int(
                    harmony_guard_crossfade_ms
                ),
                progress=progress,
                progress_points=(
                    30,
                    66,
                    71,
                    78,
                ),
                log_callback=log_callback,
            )
        )

        _progress(
            progress,
            82,
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
                82
                + int(
                    max(
                        0,
                        min(
                            100,
                            percent,
                        ),
                    )
                    * 0.08
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
            "Adaptive RVC 보컬과 변환 반주를 재합성하는 중...",
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
