from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


# FFmpeg 변환을 우선 사용하는 컨테이너/코덱 확장자.
FFMPEG_REQUIRED_EXTENSIONS = {
    ".m4a",
    ".mp4",
    ".aac",
    ".webm",
    ".mkv",
    ".mov",
    ".wma",
    ".opus",
    ".m4v",
}

# GUI에서 노출할 전체 입력 형식.
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    *FFMPEG_REQUIRED_EXTENSIONS,
}


class FFmpegNotFoundError(RuntimeError):
    pass


class FFmpegConversionError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedAudio:
    original_path: Path
    analysis_path: Path
    was_converted: bool
    temp_dir: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def find_ffmpeg() -> str | None:
    """프로젝트 로컬 FFmpeg를 우선하고, 없으면 시스템 PATH를 검색한다."""
    root = project_root()

    local_candidates = (
        root / "tools" / "ffmpeg" / "ffmpeg.exe",
        root / "tools" / "ffmpeg.exe",
        root / "ffmpeg.exe",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("ffmpeg")


def ffmpeg_status_text() -> str:
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        return f"FFmpeg 사용 가능: {ffmpeg}"
    return (
        "FFmpeg 없음: MP3/WAV/FLAC/OGG는 분석 가능하지만 "
        "M4A/MP4/AAC/WEBM/MKV/MOV 등은 SETUP_FFMPEG.bat 설치가 필요합니다."
    )


def prepare_audio_for_analysis(
    input_path: str | Path,
    *,
    sample_rate: int = 22050,
) -> PreparedAudio:
    """필요한 형식만 FFmpeg로 mono PCM WAV로 변환한다.

    WAV/FLAC/MP3/OGG 등은 기존 librosa 로딩 경로를 유지한다.
    M4A/MP4/AAC 등의 컨테이너는 FFmpeg를 통해 안정적으로 WAV로 정규화한다.
    """
    original = Path(input_path).resolve()
    if not original.is_file():
        raise FileNotFoundError(original)

    suffix = original.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        # 확장자가 낯설더라도 FFmpeg가 있다면 시도할 수 있게 한다.
        needs_ffmpeg = True
    else:
        needs_ffmpeg = suffix in FFMPEG_REQUIRED_EXTENSIONS

    if not needs_ffmpeg:
        return PreparedAudio(
            original_path=original,
            analysis_path=original,
            was_converted=False,
        )

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise FFmpegNotFoundError(
            f"{suffix or '이 형식'} 파일을 분석하려면 FFmpeg가 필요합니다.\n\n"
            "프로젝트 루트의 SETUP_FFMPEG.bat을 실행한 뒤 "
            "VS Code/터미널을 다시 열어주세요.\n\n"
            "또는 ffmpeg.exe를 다음 위치에 직접 넣어도 됩니다:\n"
            f"{project_root() / 'tools' / 'ffmpeg' / 'ffmpeg.exe'}"
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="vocal_pitch_")
    output = Path(temp_dir.name) / "analysis_input.wav"

    # -map 0:a:0 : 첫 번째 오디오 스트림 사용
    # -vn         : 영상 스트림 무시
    # pcm_s16le   : librosa/soundfile가 안정적으로 읽을 수 있는 PCM WAV
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(original),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            check=False,
        )
    except OSError as exc:
        temp_dir.cleanup()
        raise FFmpegConversionError(
            f"FFmpeg 실행에 실패했습니다: {exc}"
        ) from exc

    if completed.returncode != 0 or not output.is_file():
        error = (completed.stderr or completed.stdout or "").strip()
        temp_dir.cleanup()
        raise FFmpegConversionError(
            "FFmpeg 오디오 추출/변환에 실패했습니다.\n"
            "MP4 파일에 오디오 스트림이 없는 경우에도 이 오류가 발생할 수 있습니다.\n\n"
            f"FFmpeg 메시지:\n{error[-4000:]}"
        )

    return PreparedAudio(
        original_path=original,
        analysis_path=output,
        was_converted=True,
        temp_dir=temp_dir,
    )
