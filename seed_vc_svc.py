from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from audio_transposer import (
    AudioTransposeError,
    find_ffmpeg,
    rubberband_filter_available,
    transpose_audio,
)
from vocal_separator import (
    DEFAULT_MODEL,
    VocalSeparationError,
    _cache_key,
    _ffmpeg_env,
    _prepare_separator_input,
    _write_metadata,
    find_separator_executable,
    model_cache_dir,
    vocal_cache_root,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]

SEED_VC_REPOSITORY = "https://github.com/Plachtaa/seed-vc.git"
SEED_VC_PINNED_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"


class SeedVCSVCError(RuntimeError):
    pass


@dataclass(slots=True)
class StemPair:
    vocals_path: Path
    instrumental_path: Path
    cache_hit: bool


@dataclass(slots=True)
class ReferenceClip:
    path: Path
    start_sec: float
    duration_sec: float


def project_root() -> Path:
    return Path(__file__).resolve().parent


def seed_vc_repo_dir() -> Path:
    return project_root() / "tools" / "seed-vc"


def seed_vc_venv_dir() -> Path:
    return project_root() / ".venv_svc"


def seed_vc_python() -> Path:
    if os.name == "nt":
        return seed_vc_venv_dir() / "Scripts" / "python.exe"
    return seed_vc_venv_dir() / "bin" / "python"


def seed_vc_inference_script() -> Path:
    return seed_vc_repo_dir() / "inference.py"


def seed_vc_log_path() -> Path:
    return project_root() / "logs" / "seed_vc_svc_last.log"


def seed_vc_status_text() -> str:
    py = seed_vc_python()
    inference = seed_vc_inference_script()

    if not py.is_file() and not inference.is_file():
        return (
            "Seed-VC SVC 미설치: SETUP_SEED_VC_SVC_GPU.bat을 실행하세요."
        )

    if not py.is_file():
        return (
            "Seed-VC 저장소는 있지만 .venv_svc가 없습니다. "
            "SETUP_SEED_VC_SVC_GPU.bat을 다시 실행하세요."
        )

    if not inference.is_file():
        return (
            "Seed-VC Python 환경은 있지만 tools\\seed-vc\\inference.py가 없습니다. "
            "SETUP_SEED_VC_SVC_GPU.bat을 다시 실행하세요."
        )

    return (
        "Seed-VC SVC 사용 가능 "
        f"(pinned {SEED_VC_PINNED_COMMIT[:8]})"
    )


def seed_vc_available() -> bool:
    return (
        seed_vc_python().is_file()
        and seed_vc_inference_script().is_file()
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
            max(0, min(100, int(percent))),
            str(message),
        )


def _creationflags() -> int:
    return getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )


def _find_named_stem(
    folder: Path,
    token: str,
) -> Path | None:
    token = token.lower()

    preferred = (
        folder / f"{token}.wav",
        folder / f"{token.capitalize()}.wav",
    )

    for candidate in preferred:
        if (
            candidate.is_file()
            and candidate.stat().st_size > 0
        ):
            return candidate

    for candidate in sorted(
        folder.glob("*.wav")
    ):
        if (
            token in candidate.stem.lower()
            and candidate.stat().st_size > 0
        ):
            return candidate

    return None


def ensure_stem_pair(
    input_path: str | Path,
    *,
    model_filename: str = DEFAULT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    log_callback: LogCallback | None = None,
) -> StemPair:
    """Ensure both Vocals and Instrumental stems are available.

    The normal pitch-analysis path historically cached only vocals.wav.
    Seed-VC full-mix conversion needs the matching instrumental as well.
    If both are already cached, no separator inference is repeated.
    """
    input_file = Path(input_path).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    exe = find_separator_executable()
    if not exe:
        raise SeedVCSVCError(
            "BS-RoFormer/audio-separator가 설치되어 있지 않습니다.\n"
            "먼저 SETUP_VOCAL_SEPARATOR_GPU.bat을 실행하세요."
        )

    cache_key = _cache_key(
        input_file,
        model_filename,
    )
    final_dir = (
        vocal_cache_root() / cache_key
    )
    final_vocal = (
        final_dir / "vocals.wav"
    )
    final_instrumental = (
        final_dir / "instrumental.wav"
    )

    if (
        use_cache
        and final_vocal.is_file()
        and final_vocal.stat().st_size > 0
        and final_instrumental.is_file()
        and final_instrumental.stat().st_size > 0
    ):
        _emit(
            log_callback,
            "Seed-VC용 보컬/반주 stem 캐시를 사용합니다.",
        )
        return StemPair(
            vocals_path=final_vocal,
            instrumental_path=final_instrumental,
            cache_hit=True,
        )

    model_dir = model_cache_dir()
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_parent = (
        vocal_cache_root()
        / "_seed_vc_working"
    )
    work_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=f"{cache_key}_",
        dir=str(work_parent),
    ) as work_name:
        work_dir = Path(work_name)
        separator_temp = None

        try:
            separator_input, separator_temp = (
                _prepare_separator_input(
                    input_file,
                    log_callback=log_callback,
                )
            )

            custom_names = json.dumps(
                {
                    "Vocals": "vocals",
                    "Instrumental": "instrumental",
                },
                ensure_ascii=False,
            )

            command = [
                exe,
                str(separator_input),
                "--model_filename",
                model_filename,
                "--output_format",
                "WAV",
                "--output_dir",
                str(work_dir),
                "--model_file_dir",
                str(model_dir),
                "--sample_rate",
                "44100",
                "--custom_output_names",
                custom_names,
                "--log_level",
                "info",
            ]

            if use_autocast:
                command.append(
                    "--use_autocast"
                )

            _emit(
                log_callback,
                (
                    "Seed-VC용 2-stem 분리 시작 "
                    "(Vocals + Instrumental)"
                ),
            )

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_creationflags(),
                env=_ffmpeg_env(),
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

            if return_code != 0:
                raise SeedVCSVCError(
                    "Seed-VC용 보컬/반주 분리에 실패했습니다.\n\n"
                    + "\n".join(lines[-80:])[-7000:]
                )

            vocal = _find_named_stem(
                work_dir,
                "vocals",
            )
            instrumental = _find_named_stem(
                work_dir,
                "instrumental",
            )

            if vocal is None or instrumental is None:
                listing = "\n".join(
                    p.name
                    for p in work_dir.iterdir()
                    if p.is_file()
                )
                raise SeedVCSVCError(
                    "2-stem 분리는 끝났지만 필요한 WAV를 찾지 못했습니다.\n\n"
                    f"출력 파일:\n{listing or '(없음)'}"
                )

            if use_cache:
                final_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(
                    vocal,
                    final_vocal,
                )
                shutil.copy2(
                    instrumental,
                    final_instrumental,
                )

                _write_metadata(
                    final_dir,
                    input_path=input_file,
                    model_filename=model_filename,
                )

                _emit(
                    log_callback,
                    f"보컬 stem 캐시: {final_vocal}",
                )
                _emit(
                    log_callback,
                    f"반주 stem 캐시: {final_instrumental}",
                )

                return StemPair(
                    vocals_path=final_vocal,
                    instrumental_path=final_instrumental,
                    cache_hit=False,
                )

            # Seed-VC conversion always requests cache in the GUI.
            # This branch is retained for API completeness.
            detached = (
                project_root()
                / "cache"
                / "seed_vc"
                / "uncached_stems"
                / cache_key
            )
            detached.mkdir(
                parents=True,
                exist_ok=True,
            )
            out_vocal = (
                detached / "vocals.wav"
            )
            out_instrumental = (
                detached / "instrumental.wav"
            )
            shutil.copy2(
                vocal,
                out_vocal,
            )
            shutil.copy2(
                instrumental,
                out_instrumental,
            )

            return StemPair(
                vocals_path=out_vocal,
                instrumental_path=out_instrumental,
                cache_hit=False,
            )

        finally:
            if separator_temp is not None:
                separator_temp.cleanup()


def _block_rms_db(
    block: np.ndarray,
) -> float:
    if block.size == 0:
        return -120.0

    mono = (
        np.mean(block, axis=1)
        if block.ndim == 2
        else block
    )
    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    mono.astype(
                        np.float64,
                        copy=False,
                    )
                )
            )
        )
    )

    if (
        not math.isfinite(rms)
        or rms <= 1e-8
    ):
        return -120.0

    return 20.0 * math.log10(rms)


def create_auto_reference_clip(
    vocals_path: str | Path,
    output_path: str | Path,
    *,
    desired_seconds: float = 12.0,
    log_callback: LogCallback | None = None,
) -> ReferenceClip:
    """Pick a vocal-active reference window instead of blindly using song intro."""
    vocals_path = Path(vocals_path).resolve()
    output_path = Path(output_path).resolve()

    if not vocals_path.is_file():
        raise FileNotFoundError(vocals_path)

    block_seconds = 0.5

    with sf.SoundFile(
        str(vocals_path),
        "r",
    ) as source:
        sr = int(source.samplerate)
        total_frames = int(source.frames)
        channels = int(source.channels)

        if sr <= 0 or total_frames <= 0:
            raise SeedVCSVCError(
                "자동 참조 구간을 만들 수 없는 보컬 WAV입니다."
            )

        total_seconds = (
            total_frames / sr
        )
        desired = min(
            max(
                3.0,
                float(desired_seconds),
            ),
            total_seconds,
        )

        block_frames = max(
            1,
            int(
                round(
                    block_seconds * sr
                )
            ),
        )
        levels: list[float] = []

        source.seek(0)

        while True:
            block = source.read(
                block_frames,
                dtype="float32",
                always_2d=True,
            )
            if block.size == 0:
                break

            levels.append(
                _block_rms_db(block)
            )

        if not levels:
            raise SeedVCSVCError(
                "보컬 stem에서 참조 구간을 찾지 못했습니다."
            )

        window_blocks = max(
            1,
            int(
                round(
                    desired / block_seconds
                )
            ),
        )
        window_blocks = min(
            window_blocks,
            len(levels),
        )

        best_index = 0
        best_score = -1e9

        for start in range(
            0,
            len(levels) - window_blocks + 1,
        ):
            values = np.asarray(
                levels[
                    start:
                    start + window_blocks
                ],
                dtype=np.float64,
            )

            active = values[
                values > -48.0
            ]
            active_fraction = (
                active.size
                / max(
                    1,
                    values.size,
                )
            )

            if active.size:
                median_active = float(
                    np.median(active)
                )
                p75 = float(
                    np.percentile(
                        active,
                        75,
                    )
                )
            else:
                median_active = -120.0
                p75 = -120.0

            # Prefer long, consistently active vocal windows.
            score = (
                active_fraction * 100.0
                + median_active
                + 0.20 * p75
            )

            if score > best_score:
                best_score = score
                best_index = start

        start_sec = (
            best_index
            * block_seconds
        )
        start_frame = int(
            round(
                start_sec * sr
            )
        )
        frames_to_read = int(
            round(
                desired * sr
            )
        )

        source.seek(
            min(
                start_frame,
                max(
                    0,
                    total_frames - 1,
                ),
            )
        )

        audio = source.read(
            frames_to_read,
            dtype="float32",
            always_2d=True,
        )

    if audio.size == 0:
        raise SeedVCSVCError(
            "자동 참조 음성 추출 결과가 비어 있습니다."
        )

    mono = np.mean(
        audio,
        axis=1,
        dtype=np.float32,
    )

    # Tiny fades prevent hard cuts at reference boundaries.
    fade_frames = min(
        int(0.03 * sr),
        max(
            0,
            mono.size // 4,
        ),
    )

    if fade_frames > 0:
        fade = np.linspace(
            0.0,
            1.0,
            fade_frames,
            dtype=np.float32,
        )
        mono[:fade_frames] *= fade
        mono[-fade_frames:] *= fade[::-1]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sf.write(
        str(output_path),
        mono,
        sr,
        subtype="PCM_16",
    )

    actual_duration = (
        mono.size / sr
    )

    _emit(
        log_callback,
        (
            "같은 가수 자동 참조 구간: "
            f"{start_sec:.1f}s ~ "
            f"{start_sec + actual_duration:.1f}s "
            f"({actual_duration:.1f}s)"
        ),
    )

    return ReferenceClip(
        path=output_path,
        start_sec=start_sec,
        duration_sec=actual_duration,
    )


def _prepare_reference_file(
    reference_path: str | Path,
    output_path: str | Path,
    *,
    log_callback: LogCallback | None = None,
) -> Path:
    reference = Path(
        reference_path
    ).resolve()
    output = Path(
        output_path
    ).resolve()

    if not reference.is_file():
        raise FileNotFoundError(
            reference
        )

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SeedVCSVCError(
            "참조 음성을 WAV로 정규화하려면 FFmpeg가 필요합니다."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(reference),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creationflags(),
        check=False,
    )

    if (
        completed.returncode != 0
        or not output.is_file()
    ):
        raise SeedVCSVCError(
            "참조 음성 변환에 실패했습니다.\n\n"
            + (
                completed.stderr
                or completed.stdout
                or ""
            )[-5000:]
        )

    _emit(
        log_callback,
        f"사용자 참조 음성 준비: {reference.name}",
    )

    return output


def _run_seed_vc(
    source_vocal: Path,
    reference_wav: Path,
    output_dir: Path,
    *,
    semitones: int,
    diffusion_steps: int,
    cfg_rate: float,
    fp16: bool,
    log_callback: LogCallback | None = None,
) -> Path:
    if not seed_vc_available():
        raise SeedVCSVCError(
            seed_vc_status_text()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    before = {
        p.resolve()
        for p in output_dir.glob(
            "*.wav"
        )
    }

    command = [
        str(seed_vc_python()),
        str(seed_vc_inference_script()),
        "--source",
        str(source_vocal),
        "--target",
        str(reference_wav),
        "--output",
        str(output_dir),
        "--diffusion-steps",
        str(int(diffusion_steps)),
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        f"{float(cfg_rate):.3f}",
        "--f0-condition",
        "True",
        "--auto-f0-adjust",
        "False",
        "--semi-tone-shift",
        str(int(semitones)),
        "--fp16",
        "True"
        if fp16
        else "False",
    ]

    env = os.environ.copy()
    env.setdefault(
        "PYTHONUTF8",
        "1",
    )
    env.setdefault(
        "PYTHONIOENCODING",
        "utf-8",
    )
    env.setdefault(
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "1",
    )

    lines: list[str] = []

    _emit(
        log_callback,
        (
            "Seed-VC SVC 시작: "
            f"{semitones:+d} semitone / "
            f"{diffusion_steps} steps / "
            f"CFG {cfg_rate:.2f} / "
            f"FP16 {'ON' if fp16 else 'OFF'}"
        ),
    )

    process = subprocess.Popen(
        command,
        cwd=str(
            seed_vc_repo_dir()
        ),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_creationflags(),
    )

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

    log_path = seed_vc_log_path()
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        log_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    if return_code != 0:
        raise SeedVCSVCError(
            "Seed-VC SVC 추론에 실패했습니다.\n\n"
            f"로그: {log_path}\n\n"
            + "\n".join(
                lines[-100:]
            )[-8000:]
        )

    candidates = [
        p
        for p in output_dir.glob(
            "*.wav"
        )
        if p.resolve() not in before
        and p.is_file()
        and p.stat().st_size > 0
    ]

    if not candidates:
        candidates = [
            p
            for p in output_dir.glob(
                "*.wav"
            )
            if p.is_file()
            and p.stat().st_size > 0
        ]

    if not candidates:
        raise SeedVCSVCError(
            "Seed-VC는 종료됐지만 출력 WAV를 찾지 못했습니다."
        )

    candidates.sort(
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )

    result = candidates[0]

    _emit(
        log_callback,
        f"Seed-VC 보컬 생성 완료: {result.name}",
    )

    return result


def _audio_rms(
    path: Path,
) -> float:
    sum_squares = 0.0
    sample_count = 0

    with sf.SoundFile(
        str(path),
        "r",
    ) as audio:
        while True:
            block = audio.read(
                65536,
                dtype="float32",
                always_2d=True,
            )

            if block.size == 0:
                break

            mono = np.mean(
                block,
                axis=1,
            ).astype(
                np.float64,
                copy=False,
            )

            sum_squares += float(
                np.sum(
                    mono * mono
                )
            )
            sample_count += (
                mono.size
            )

    if sample_count <= 0:
        return 0.0

    return math.sqrt(
        sum_squares
        / sample_count
    )


def _vocal_gain_for_mix(
    source_vocal: Path,
    generated_vocal: Path,
) -> float:
    source_rms = _audio_rms(
        source_vocal
    )
    generated_rms = _audio_rms(
        generated_vocal
    )

    if (
        source_rms <= 1e-8
        or generated_rms <= 1e-8
    ):
        return 1.0

    gain = (
        source_rms
        / generated_rms
    )

    # Avoid an extreme gain caused by a bad/silent reference.
    return float(
        min(
            2.0,
            max(
                0.5,
                gain,
            ),
        )
    )


def _codec_args(
    output_path: Path,
) -> list[str]:
    suffix = (
        output_path.suffix.lower()
    )

    if suffix == ".wav":
        return [
            "-c:a",
            "pcm_s24le",
        ]

    if suffix == ".flac":
        return [
            "-c:a",
            "flac",
            "-compression_level",
            "8",
        ]

    if suffix == ".mp3":
        return [
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
        ]

    if suffix == ".m4a":
        return [
            "-c:a",
            "aac",
            "-b:a",
            "256k",
        ]

    raise SeedVCSVCError(
        "지원하지 않는 출력 형식: "
        f"{suffix or '(없음)'}"
    )


def _encode_single_vocal(
    source_wav: Path,
    output_path: Path,
) -> None:
    ffmpeg = find_ffmpeg()

    if not ffmpeg:
        raise SeedVCSVCError(
            "출력 인코딩에 필요한 FFmpeg를 찾지 못했습니다."
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_wav),
        "-map",
        "0:a:0",
        "-vn",
    ]
    command.extend(
        _codec_args(
            output_path
        )
    )
    command.append(
        str(output_path)
    )

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creationflags(),
        check=False,
    )

    if (
        completed.returncode != 0
        or not output_path.is_file()
    ):
        raise SeedVCSVCError(
            "Seed-VC 보컬 출력 인코딩에 실패했습니다.\n\n"
            + (
                completed.stderr
                or completed.stdout
                or ""
            )[-5000:]
        )


def _mix_stems(
    generated_vocal: Path,
    shifted_instrumental: Path,
    original_vocal: Path,
    output_path: Path,
    *,
    log_callback: LogCallback | None = None,
) -> None:
    ffmpeg = find_ffmpeg()

    if not ffmpeg:
        raise SeedVCSVCError(
            "보컬/반주 재합성에 필요한 FFmpeg를 찾지 못했습니다."
        )

    gain = _vocal_gain_for_mix(
        original_vocal,
        generated_vocal,
    )
    gain_db = (
        20.0 * math.log10(gain)
        if gain > 0
        else -120.0
    )

    _emit(
        log_callback,
        (
            "Seed-VC 보컬 레벨 보정: "
            f"{gain:.3f}x ({gain_db:+.2f} dB)"
        ),
    )

    filter_complex = (
        f"[0:a]volume={gain:.8f}[v];"
        "[v][1:a]"
        "amix=inputs=2:duration=longest:normalize=0,"
        "alimiter=limit=0.97"
        "[mix]"
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(generated_vocal),
        "-i",
        str(shifted_instrumental),
        "-filter_complex",
        filter_complex,
        "-map",
        "[mix]",
        "-vn",
    ]
    command.extend(
        _codec_args(
            output_path
        )
    )
    command.append(
        str(output_path)
    )

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creationflags(),
        check=False,
    )

    if (
        completed.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size <= 0
    ):
        raise SeedVCSVCError(
            "Seed-VC 보컬과 변환 반주 재합성에 실패했습니다.\n\n"
            + (
                completed.stderr
                or completed.stdout
                or ""
            )[-5000:]
        )


def convert_vocal_seed_vc(
    vocal_path: str | Path,
    output_path: str | Path,
    *,
    semitones: int,
    reference_path: str | Path | None = None,
    auto_reference: bool = True,
    diffusion_steps: int = 30,
    cfg_rate: float = 0.7,
    fp16: bool = True,
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

    if not seed_vc_available():
        raise SeedVCSVCError(
            seed_vc_status_text()
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="vocal_pitch_seed_vc_"
    ) as temp_name:
        temp_dir = Path(
            temp_name
        )

        _progress(
            progress,
            10,
            "Seed-VC 참조 음성을 준비하는 중...",
        )

        if auto_reference:
            reference = create_auto_reference_clip(
                vocal,
                temp_dir / "reference_auto.wav",
                desired_seconds=12.0,
                log_callback=log_callback,
            ).path
        else:
            if reference_path is None:
                raise SeedVCSVCError(
                    "사용자 참조 음성 파일이 선택되지 않았습니다."
                )

            reference = _prepare_reference_file(
                reference_path,
                temp_dir / "reference_custom.wav",
                log_callback=log_callback,
            )

        _progress(
            progress,
            20,
            (
                "Seed-VC SVC 추론 중... "
                "첫 실행이면 AI 모델을 다운로드합니다."
            ),
        )

        generated = _run_seed_vc(
            vocal,
            reference,
            temp_dir / "seed_output",
            semitones=semitones,
            diffusion_steps=diffusion_steps,
            cfg_rate=cfg_rate,
            fp16=fp16,
            log_callback=log_callback,
        )

        _progress(
            progress,
            90,
            "Seed-VC 보컬을 출력 형식으로 인코딩하는 중...",
        )

        _encode_single_vocal(
            generated,
            output,
        )

    _progress(
        progress,
        100,
        f"Seed-VC 보컬 변환 완료: {output.name}",
    )

    return output


def convert_full_mix_seed_vc(
    input_path: str | Path,
    output_path: str | Path,
    *,
    semitones: int,
    reference_path: str | Path | None = None,
    auto_reference: bool = True,
    diffusion_steps: int = 30,
    cfg_rate: float = 0.7,
    fp16: bool = True,
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

    if not seed_vc_available():
        raise SeedVCSVCError(
            seed_vc_status_text()
        )

    available, rb_status = (
        rubberband_filter_available()
    )
    if not available:
        raise SeedVCSVCError(
            "원곡 전체 AI 변환은 반주도 같은 키로 이동해야 합니다.\n"
            f"{rb_status}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _progress(
        progress,
        3,
        "Seed-VC용 보컬/반주 stem을 준비하는 중...",
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
        prefix="vocal_pitch_seed_vc_mix_"
    ) as temp_name:
        temp_dir = Path(
            temp_name
        )

        if auto_reference:
            reference = create_auto_reference_clip(
                pair.vocals_path,
                temp_dir / "reference_auto.wav",
                desired_seconds=12.0,
                log_callback=log_callback,
            ).path
        else:
            if reference_path is None:
                raise SeedVCSVCError(
                    "사용자 참조 음성 파일이 선택되지 않았습니다."
                )

            reference = _prepare_reference_file(
                reference_path,
                temp_dir / "reference_custom.wav",
                log_callback=log_callback,
            )

        _progress(
            progress,
            32,
            (
                "Seed-VC SVC 보컬 변환 중... "
                "첫 실행이면 모델 다운로드로 오래 걸릴 수 있습니다."
            ),
        )

        generated_vocal = _run_seed_vc(
            pair.vocals_path,
            reference,
            temp_dir / "seed_output",
            semitones=semitones,
            diffusion_steps=diffusion_steps,
            cfg_rate=cfg_rate,
            fp16=fp16,
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
            raise SeedVCSVCError(
                "Seed-VC 보컬은 생성됐지만 반주 키 변경에 실패했습니다.\n\n"
                f"{exc}"
            ) from exc

        _progress(
            progress,
            92,
            "AI 보컬과 변환된 반주를 재합성하는 중...",
        )

        _mix_stems(
            generated_vocal,
            shifted_instrumental,
            pair.vocals_path,
            output,
            log_callback=log_callback,
        )

    _progress(
        progress,
        100,
        f"Seed-VC AI 키 변환 완료: {output.name}",
    )

    return output
