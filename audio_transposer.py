from __future__ import annotations

from pathlib import Path
from typing import Callable
import os
import shutil
import subprocess


ProgressCallback = Callable[[int, str], None]


class AudioTransposeError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_ffmpeg() -> str | None:
    root = project_root()

    candidates = (
        root / "tools" / "ffmpeg" / "ffmpeg.exe",
        root / "tools" / "ffmpeg.exe",
        root / "ffmpeg.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("ffmpeg")


def find_ffprobe() -> str | None:
    ffmpeg = find_ffmpeg()

    if ffmpeg:
        sibling = Path(ffmpeg).with_name(
            "ffprobe.exe"
            if os.name == "nt"
            else "ffprobe"
        )
        if sibling.is_file():
            return str(sibling)

    root = project_root()
    candidates = (
        root / "tools" / "ffmpeg" / "ffprobe.exe",
        root / "tools" / "ffprobe.exe",
        root / "ffprobe.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("ffprobe")


def semitone_to_ratio(semitones: float) -> float:
    return 2.0 ** (float(semitones) / 12.0)


def rubberband_filter_available() -> tuple[bool, str]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False, "FFmpeg를 찾을 수 없습니다."

    creationflags = getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )

    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-filters",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )

    if completed.returncode != 0:
        return (
            False,
            "FFmpeg 필터 목록 조회에 실패했습니다.",
        )

    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "rubberband":
            return True, "FFmpeg rubberband 사용 가능"

    return (
        False,
        "현재 FFmpeg 빌드에 rubberband 필터가 없습니다.",
    )


def get_duration_seconds(
    input_path: str | Path,
) -> float | None:
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None

    creationflags = getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )

    if completed.returncode != 0:
        return None

    try:
        value = float(
            completed.stdout.strip()
        )
    except ValueError:
        return None

    return value if value > 0 else None


def _filter_expression(
    *,
    semitones: float,
    preserve_formant: bool,
    quality: str,
) -> str:
    ratio = semitone_to_ratio(semitones)

    quality = str(quality).lower()
    if quality not in {
        "quality",
        "speed",
        "consistency",
    }:
        quality = "quality"

    options = [
        "tempo=1",
        f"pitch={ratio:.12f}",
        f"pitchq={quality}",
    ]

    if preserve_formant:
        options.append(
            "formant=preserved"
        )
    else:
        options.append(
            "formant=shifted"
        )

    return (
        "rubberband="
        + ":".join(options)
    )


def _codec_args_for_output(
    output_path: Path,
) -> list[str]:
    suffix = output_path.suffix.lower()

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

    if suffix in {".m4a", ".mp4"}:
        return [
            "-c:a",
            "aac",
            "-b:a",
            "256k",
        ]

    raise AudioTransposeError(
        "지원하지 않는 출력 확장자입니다: "
        f"{suffix or '(없음)'}"
    )


def transpose_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    semitones: float,
    preserve_formant: bool = True,
    quality: str = "quality",
    progress: ProgressCallback | None = None,
) -> Path:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AudioTransposeError(
            "FFmpeg를 찾을 수 없습니다."
        )

    available, status = (
        rubberband_filter_available()
    )
    if not available:
        raise AudioTransposeError(
            status
            + "\n\n"
            "키 변경은 FFmpeg의 rubberband 필터가 필요합니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duration = get_duration_seconds(
        input_path
    )

    filter_expr = _filter_expression(
        semitones=semitones,
        preserve_formant=preserve_formant,
        quality=quality,
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        filter_expr,
        "-progress",
        "pipe:1",
        "-nostats",
    ]
    command.extend(
        _codec_args_for_output(
            output_path
        )
    )
    command.append(
        str(output_path)
    )

    creationflags = getattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0,
    )

    if progress:
        progress(
            0,
            (
                f"키 변경 시작: {semitones:+.0f} semitone "
                f"(ratio={semitone_to_ratio(semitones):.4f})"
            ),
        )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )

    assert process.stdout is not None

    last_percent = -1

    for raw_line in process.stdout:
        line = raw_line.strip()

        if not line.startswith(
            "out_time_us="
        ):
            continue

        try:
            out_us = int(
                line.split(
                    "=",
                    1,
                )[1]
            )
        except ValueError:
            continue

        if (
            duration is not None
            and duration > 0
        ):
            current = (
                out_us / 1_000_000.0
            )
            percent = min(
                99,
                max(
                    0,
                    int(
                        current
                        / duration
                        * 100.0
                    ),
                ),
            )

            if (
                progress
                and percent
                != last_percent
            ):
                progress(
                    percent,
                    (
                        f"키 변경 중... "
                        f"{current:.1f}/{duration:.1f}초"
                    ),
                )
                last_percent = percent

    stderr_text = (
        process.stderr.read()
        if process.stderr is not None
        else ""
    )

    return_code = process.wait()

    if return_code != 0:
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass

        raise AudioTransposeError(
            "FFmpeg 키 변경에 실패했습니다.\n\n"
            + stderr_text[-7000:]
        )

    if (
        not output_path.is_file()
        or output_path.stat().st_size <= 0
    ):
        raise AudioTransposeError(
            "FFmpeg는 종료됐지만 출력 파일이 생성되지 않았습니다."
        )

    if progress:
        progress(
            100,
            f"키 변경 완료: {output_path.name}",
        )

    return output_path
