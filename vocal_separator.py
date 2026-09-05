from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time


DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

SEPARATOR_PRECONVERT_EXTENSIONS = {
    ".m4a",
    ".mp4",
    ".aac",
    ".webm",
    ".mkv",
    ".mov",
    ".wma",
    ".m4v",
}

LogCallback = Callable[[str], None]


class VocalSeparatorNotInstalledError(RuntimeError):
    pass


class VocalSeparationError(RuntimeError):
    pass


@dataclass(slots=True)
class SeparatedVocal:
    original_path: Path
    vocal_path: Path
    model_filename: str
    cache_hit: bool
    cached: bool
    temp_dir: tempfile.TemporaryDirectory | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def separator_venv_dir() -> Path:
    return project_root() / ".venv_separator"


def model_cache_dir() -> Path:
    return project_root() / "cache" / "separator_models"


def vocal_cache_root() -> Path:
    return project_root() / "cache" / "vocal_stems"


def separator_log_dir() -> Path:
    return project_root() / "logs"


def separator_log_path() -> Path:
    return separator_log_dir() / "vocal_separator_last.log"


def find_separator_executable() -> str | None:
    root = project_root()
    candidates = [
        root / ".venv_separator" / "Scripts" / "audio-separator.exe",
        root / ".venv_separator" / "Scripts" / "audio-separator",
        root / ".venv" / "Scripts" / "audio-separator.exe",
        root / ".venv" / "Scripts" / "audio-separator",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return shutil.which("audio-separator")


def separator_status_text() -> str:
    exe = find_separator_executable()
    if exe:
        return f"보컬 분리기 사용 가능: {exe}"
    return (
        "보컬 분리기 미설치: SETUP_VOCAL_SEPARATOR_GPU.bat을 실행하세요. "
        "원본 분석은 계속 사용할 수 있습니다."
    )


def _ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    root = project_root()

    local_dirs = []
    if (root / "tools" / "ffmpeg" / "ffmpeg.exe").is_file():
        local_dirs.append(str(root / "tools" / "ffmpeg"))
    if (root / "tools" / "ffmpeg.exe").is_file():
        local_dirs.append(str(root / "tools"))
    if (root / "ffmpeg.exe").is_file():
        local_dirs.append(str(root))

    if local_dirs:
        env["PATH"] = os.pathsep.join(local_dirs + [env.get("PATH", "")])

    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _find_ffmpeg_for_separator() -> str | None:
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


def _prepare_separator_input(
    input_file: Path,
    log_callback: LogCallback | None = None,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    # libsndfile가 MP4/M4A 컨테이너를 직접 못 읽는 환경을 위해
    # 분리 전에 stereo 44.1kHz lossless PCM WAV로 정규화한다.
    suffix = input_file.suffix.lower()

    if suffix not in SEPARATOR_PRECONVERT_EXTENSIONS:
        return input_file, None

    ffmpeg = _find_ffmpeg_for_separator()
    if not ffmpeg:
        raise VocalSeparationError(
            f"{suffix} 입력을 보컬 분리하려면 FFmpeg가 필요합니다.\n"
            "SETUP_FFMPEG.bat / CHECK_FFMPEG.bat을 확인하세요."
        )

    temp_dir = tempfile.TemporaryDirectory(
        prefix="vocal_pitch_separator_input_"
    )
    output = Path(temp_dir.name) / "separator_input.wav"

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]

    if log_callback:
        log_callback(
            f"FFmpeg 분리용 변환: {input_file.suffix or '(unknown)'} "
            "-> stereo 44.1kHz PCM WAV"
        )

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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

    if completed.returncode != 0 or not output.is_file():
        error = (completed.stderr or completed.stdout or "").strip()
        temp_dir.cleanup()
        raise VocalSeparationError(
            "FFmpeg 분리용 입력 변환에 실패했습니다.\n\n"
            f"{error[-5000:]}"
        )

    if log_callback:
        log_callback(f"분리용 WAV 준비 완료: {output}")

    return output, temp_dir


def _cache_key(input_path: Path, model_filename: str) -> str:
    stat = input_path.stat()
    payload = {
        "path": str(input_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "model": model_filename,
        "cache_schema": 2,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _write_metadata(
    folder: Path,
    *,
    input_path: Path,
    model_filename: str,
) -> None:
    data = {
        "source": str(input_path.resolve()),
        "source_size": input_path.stat().st_size,
        "source_mtime_ns": input_path.stat().st_mtime_ns,
        "model_filename": model_filename,
        "created_unix": time.time(),
        "separator_input_pipeline": "ffmpeg_stereo_44100_pcm_when_needed",
    }
    (folder / "metadata.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_vocal_output(output_dir: Path) -> Path | None:
    preferred = [
        output_dir / "vocals.wav",
        output_dir / "Vocals.wav",
        output_dir / "vocals.WAV",
    ]
    for path in preferred:
        if path.is_file() and path.stat().st_size > 0:
            return path

    wavs = sorted(output_dir.glob("*.wav"))
    vocal_named = [
        p for p in wavs
        if "vocal" in p.stem.lower()
    ]
    for path in vocal_named:
        if path.stat().st_size > 0:
            return path

    valid_wavs = [p for p in wavs if p.stat().st_size > 0]
    if len(valid_wavs) == 1:
        return valid_wavs[0]

    return None


def separate_vocals(
    input_path: str | Path,
    *,
    model_filename: str = DEFAULT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    log_callback: LogCallback | None = None,
) -> SeparatedVocal:
    input_file = Path(input_path).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    exe = find_separator_executable()
    if not exe:
        raise VocalSeparatorNotInstalledError(
            "AI 보컬 분리기가 설치되어 있지 않습니다.\n\n"
            "프로젝트 루트에서 SETUP_VOCAL_SEPARATOR_GPU.bat을 실행하세요.\n"
            "설치 후 CHECK_VOCAL_SEPARATOR.bat으로 환경을 확인할 수 있습니다."
        )

    model_dir = model_cache_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    log_dir = separator_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = separator_log_path()

    cache_key = _cache_key(input_file, model_filename)

    if use_cache:
        final_dir = vocal_cache_root() / cache_key
        final_vocal = final_dir / "vocals.wav"

        if final_vocal.is_file() and final_vocal.stat().st_size > 0:
            if log_callback:
                log_callback("기존 보컬 분리 캐시를 사용합니다.")
            return SeparatedVocal(
                original_path=input_file,
                vocal_path=final_vocal,
                model_filename=model_filename,
                cache_hit=True,
                cached=True,
                temp_dir=None,
            )

        work_parent = vocal_cache_root() / "_working"
        work_parent.mkdir(parents=True, exist_ok=True)
        temp_obj = tempfile.TemporaryDirectory(
            prefix=f"{cache_key}_",
            dir=str(work_parent),
        )
    else:
        temp_obj = tempfile.TemporaryDirectory(
            prefix="vocal_pitch_separator_"
        )

    output_dir = Path(temp_obj.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator_input_temp: tempfile.TemporaryDirectory | None = None

    try:
        separator_input, separator_input_temp = _prepare_separator_input(
            input_file,
            log_callback=log_callback,
        )

        custom_names = json.dumps(
            {"Vocals": "vocals"},
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
            str(output_dir),
            "--model_file_dir",
            str(model_dir),
            "--single_stem",
            "Vocals",
            "--sample_rate",
            "44100",
            "--custom_output_names",
            custom_names,
            "--log_level",
            "info",
        ]

        if use_autocast:
            command.append("--use_autocast")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        lines: list[str] = []

        def emit(message: str) -> None:
            clean = message.strip()
            if not clean:
                return
            lines.append(clean)
            if log_callback:
                log_callback(clean)

        emit(f"보컬 분리 시작: {input_file.name}")
        emit(f"모델: {model_filename}")
        emit(f"실행기: {exe}")
        if separator_input != input_file:
            emit(f"실제 separator 입력: {separator_input}")
        emit("첫 실행이면 모델 다운로드 때문에 시간이 더 걸릴 수 있습니다.")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=_ffmpeg_env(),
            )
        except OSError as exc:
            raise VocalSeparationError(
                f"audio-separator 실행에 실패했습니다: {exc}"
            ) from exc

        assert process.stdout is not None

        try:
            for raw_line in process.stdout:
                for part in raw_line.replace("\r", "\n").splitlines():
                    if part.strip():
                        emit(part)

            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise

        try:
            log_file.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

        if return_code != 0:
            error_tail = "\n".join(lines[-80:])
            raise VocalSeparationError(
                "AI 보컬 분리에 실패했습니다.\n\n"
                f"로그 파일: {log_file}\n\n"
                f"마지막 로그:\n{error_tail[-7000:]}"
            )

        produced = _find_vocal_output(output_dir)
        if produced is None:
            listing = "\n".join(
                p.name for p in output_dir.iterdir()
                if p.is_file()
            )
            raise VocalSeparationError(
                "분리 프로세스는 종료됐지만 vocals WAV를 찾지 못했습니다.\n\n"
                f"출력 폴더 파일:\n{listing or '(없음)'}\n\n"
                f"로그: {log_file}"
            )

        if use_cache:
            final_dir = vocal_cache_root() / cache_key
            final_dir.mkdir(parents=True, exist_ok=True)
            final_vocal = final_dir / "vocals.wav"

            shutil.copy2(produced, final_vocal)
            _write_metadata(
                final_dir,
                input_path=input_file,
                model_filename=model_filename,
            )

            emit(f"보컬 캐시 저장: {final_vocal}")
            return SeparatedVocal(
                original_path=input_file,
                vocal_path=final_vocal,
                model_filename=model_filename,
                cache_hit=False,
                cached=True,
                temp_dir=None,
            )

        emit(f"임시 보컬 stem 생성: {produced}")
        return SeparatedVocal(
            original_path=input_file,
            vocal_path=produced,
            model_filename=model_filename,
            cache_hit=False,
            cached=False,
            temp_dir=temp_obj,
        )

    except Exception:
        if use_cache:
            temp_obj.cleanup()
        raise

    finally:
        if separator_input_temp is not None:
            separator_input_temp.cleanup()
