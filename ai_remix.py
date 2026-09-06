from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
import http.client
import mimetypes
import secrets
import subprocess
import tempfile

from media_input import find_ffmpeg
import json
import os
import shutil
import subprocess
import time


# V31_ACESTEP_SRC_AUDIO_UPLOAD_HOTFIX
# V31_AI_REMIX_ACESTEP_PATCH

ACE_REPOSITORY = "https://github.com/ace-step/ACE-Step-1.5.git"
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8001
DEFAULT_DIT_MODEL = "acestep-v15-turbo"
DEFAULT_LM_MODEL = "acestep-5Hz-lm-0.6B"


STYLE_PRESETS: dict[str, dict[str, str]] = {
    "blues": {
        "label": "Blues / 블루스",
        "prompt": (
            "vintage electric blues, expressive blues guitar, "
            "warm tube amplifier, soulful organ, laid-back live drums, "
            "deep electric bass, smoky club atmosphere, organic human groove"
        ),
    },
    "korean_7080": {
        "label": "7080 한국 가요",
        "prompt": (
            "1970s to 1980s Korean popular music arrangement, "
            "warm analog tape sound, vintage electric piano, acoustic guitar, "
            "lush strings, melodic bass, restrained live drums, nostalgic emotional mood"
        ),
    },
    "ballad": {
        "label": "Modern Ballad / 발라드",
        "prompt": (
            "emotional modern Korean ballad, intimate piano, cinematic strings, "
            "soft live drums, warm bass, gradual dynamic build, spacious polished mix"
        ),
    },
    "rnb": {
        "label": "R&B / 알앤비",
        "prompt": (
            "contemporary R&B, warm electric piano, neo soul chords, "
            "syncopated drums, deep smooth bass, atmospheric pads, "
            "subtle guitar fills, polished intimate production"
        ),
    },
    "jazz": {
        "label": "Jazz / 재즈",
        "prompt": (
            "small jazz ensemble, acoustic piano, upright bass, brushed drums, "
            "warm saxophone and guitar comping, sophisticated reharmonization, "
            "natural room ambience"
        ),
    },
    "citypop": {
        "label": "City Pop / 시티팝",
        "prompt": (
            "1980s Japanese city pop inspired arrangement, bright electric piano, "
            "funk guitar, melodic bass, crisp drums, analog synthesizers, "
            "glossy nostalgic night-drive atmosphere"
        ),
    },
    "rock": {
        "label": "Rock / 록",
        "prompt": (
            "energetic live rock band, crunchy electric guitars, punchy acoustic drums, "
            "driving electric bass, dynamic build, wide chorus, organic performance"
        ),
    },
    "acoustic": {
        "label": "Acoustic / 어쿠스틱",
        "prompt": (
            "intimate acoustic arrangement, fingerstyle acoustic guitar, soft piano, "
            "minimal percussion, warm upright bass, natural room, emotional and organic"
        ),
    },
    "orchestral": {
        "label": "Orchestral / 오케스트라",
        "prompt": (
            "cinematic orchestral arrangement, expressive strings, piano, "
            "woodwinds, brass swells, timpani and restrained percussion, "
            "wide dynamic range, emotional film score production"
        ),
    },
    "custom": {
        "label": "사용자 지정",
        "prompt": "",
    },
}


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


class AIRemixError(RuntimeError):
    pass


class AIRemixCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class AIRemixResult:
    source_path: Path
    output_path: Path
    task_id: str
    style_key: str
    prompt: str
    cover_strength: float
    seed: int | None
    metadata: dict


_SERVER_PROCESS: subprocess.Popen | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def ace_repo_dir() -> Path:
    return (
        project_root()
        / "tools"
        / "ACE-Step-1.5"
    )


def ace_python() -> Path:
    return (
        project_root()
        / ".venv_remix"
        / "Scripts"
        / "python.exe"
    )


def logs_dir() -> Path:
    return (
        project_root()
        / "logs"
    )


def remix_log_path() -> Path:
    return (
        logs_dir()
        / "ai_remix_last.log"
    )


def remix_json_path() -> Path:
    return (
        logs_dir()
        / "ai_remix_last.json"
    )


def server_log_path() -> Path:
    return (
        logs_dir()
        / "ace_step_server.log"
    )


def api_base_url(
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
) -> str:
    return f"http://{host}:{int(port)}"


def ace_step_available() -> bool:
    return bool(
        ace_python().is_file()
        and (
            ace_repo_dir()
            / "acestep"
            / "api_server.py"
        ).is_file()
    )


def ace_step_status_text() -> str:
    if ace_step_available():
        return (
            "ACE-Step 1.5 로컬 환경 준비됨 "
            f"({ace_python()})"
        )

    missing: list[str] = []

    if not ace_python().is_file():
        missing.append(
            ".venv_remix"
        )

    if not (
        ace_repo_dir()
        / "acestep"
        / "api_server.py"
    ).is_file():
        missing.append(
            "tools\\ACE-Step-1.5"
        )

    return (
        "ACE-Step 1.5 미설치: "
        + ", ".join(
            missing
        )
        + " / SETUP_AI_REMIX_ACESTEP.bat 실행 필요"
    )


def style_prompt(
    style_key: str,
    custom_prompt: str = "",
) -> str:
    key = (
        style_key
        if style_key
        in STYLE_PRESETS
        else "custom"
    )

    preset = str(
        STYLE_PRESETS[
            key
        ][
            "prompt"
        ]
    ).strip()

    extra = str(
        custom_prompt
        or ""
    ).strip()

    if key == "custom":
        return extra

    if extra:
        return (
            preset
            + ", "
            + extra
        )

    return preset


def _emit(
    lines: list[str],
    callback: LogCallback | None,
    message: str,
) -> None:
    clean = str(
        message
    ).strip()

    if not clean:
        return

    lines.append(
        clean
    )

    if callback is not None:
        callback(
            clean
        )


def _progress(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is None:
        return

    callback(
        max(
            0,
            min(
                100,
                int(
                    percent
                ),
            ),
        ),
        str(
            message
        ),
    )


def _write_logs(
    lines: list[str],
    payload: dict,
) -> None:
    try:
        logs_dir().mkdir(
            parents=True,
            exist_ok=True,
        )
        remix_log_path().write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    try:
        remix_json_path().write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    *,
    timeout: float = 20.0,
) -> dict:
    body = None
    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
        headers[
            "Content-Type"
        ] = "application/json"

    request = urlrequest.Request(
        url,
        data=body,
        headers=headers,
        method=method.upper(),
    )

    try:
        with urlrequest.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read()
    except urlerror.HTTPError as exc:
        try:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            detail = str(
                exc
            )

        raise AIRemixError(
            f"ACE-Step API HTTP {exc.code}: {detail}"
        ) from exc

    except urlerror.URLError as exc:
        raise AIRemixError(
            f"ACE-Step API 연결 실패: {exc}"
        ) from exc

    try:
        return json.loads(
            raw.decode(
                "utf-8"
            )
        )
    except Exception as exc:
        raise AIRemixError(
            "ACE-Step API가 JSON이 아닌 응답을 반환했습니다."
        ) from exc


def api_health(
    *,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    timeout: float = 2.0,
) -> bool:
    url = (
        api_base_url(
            host,
            port,
        )
        + "/health"
    )

    try:
        _http_json(
            "GET",
            url,
            timeout=timeout,
        )
        return True
    except Exception:
        return False


def start_api_server(
    *,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    log_callback: LogCallback | None = None,
) -> subprocess.Popen | None:
    global _SERVER_PROCESS

    if api_health(
        host=host,
        port=port,
    ):
        if log_callback is not None:
            log_callback(
                "[ACE-Step] 기존 API 서버 사용"
            )
        return _SERVER_PROCESS

    if not ace_step_available():
        raise AIRemixError(
            ace_step_status_text()
        )

    if (
        _SERVER_PROCESS is not None
        and _SERVER_PROCESS.poll()
        is None
    ):
        return _SERVER_PROCESS

    logs_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()
    env[
        "ACESTEP_API_HOST"
    ] = str(
        host
    )
    env[
        "ACESTEP_API_PORT"
    ] = str(
        int(
            port
        )
    )
    env[
        "ACESTEP_API_WORKERS"
    ] = "1"
    env[
        "ACESTEP_CONFIG_PATH"
    ] = DEFAULT_DIT_MODEL
    env[
        "ACESTEP_LM_MODEL_PATH"
    ] = DEFAULT_LM_MODEL
    env[
        "ACESTEP_DEVICE"
    ] = "auto"
    env[
        "ACESTEP_INIT_LLM"
    ] = "auto"
    env[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    log_handle = open(
        server_log_path(),
        "a",
        encoding="utf-8",
    )

    creationflags = (
        getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        )
        | getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    )

    command = [
        str(
            ace_python()
        ),
        str(
            ace_repo_dir()
            / "acestep"
            / "api_server.py"
        ),
    ]

    if log_callback is not None:
        log_callback(
            "[ACE-Step] 로컬 API 서버 시작"
        )
        log_callback(
            "[ACE-Step] 첫 실행이면 모델 다운로드/로딩으로 수 분 걸릴 수 있습니다."
        )
        log_callback(
            f"[ACE-Step] 서버 로그: {server_log_path()}"
        )

    try:
        _SERVER_PROCESS = subprocess.Popen(
            command,
            cwd=str(
                ace_repo_dir()
            ),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    except Exception:
        log_handle.close()
        raise

    return _SERVER_PROCESS


def ensure_api_server(
    *,
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    timeout_seconds: float = 900.0,
    log_callback: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> None:
    if api_health(
        host=host,
        port=port,
    ):
        _progress(
            progress,
            10,
            "ACE-Step API 준비됨",
        )
        return

    process = start_api_server(
        host=host,
        port=port,
        log_callback=log_callback,
    )

    started = time.monotonic()
    last_notice = -1

    while (
        time.monotonic()
        - started
        < float(
            timeout_seconds
        )
    ):
        if (
            cancel_check is not None
            and cancel_check()
        ):
            raise AIRemixCancelled(
                "AI 리믹스 준비 대기를 중지했습니다."
            )

        if api_health(
            host=host,
            port=port,
        ):
            _progress(
                progress,
                15,
                "ACE-Step 모델/API 준비 완료",
            )

            if log_callback is not None:
                log_callback(
                    "[ACE-Step] API 준비 완료"
                )

            return

        if (
            process is not None
            and process.poll()
            is not None
        ):
            raise AIRemixError(
                "ACE-Step API 서버가 준비되기 전에 종료됐습니다.\n"
                f"로그 확인: {server_log_path()}"
            )

        elapsed = int(
            time.monotonic()
            - started
        )

        notice = (
            elapsed
            // 15
        )

        if notice != last_notice:
            last_notice = notice
            _progress(
                progress,
                min(
                    14,
                    3
                    + notice,
                ),
                (
                    "ACE-Step 모델 로딩 중... "
                    f"{elapsed}s"
                ),
            )

        time.sleep(
            2.0
        )

    raise AIRemixError(
        "ACE-Step API 준비 시간이 15분을 초과했습니다.\n"
        f"로그 확인: {server_log_path()}"
    )


def _unwrap_response(
    payload: dict,
) -> object:
    if not isinstance(
        payload,
        dict,
    ):
        raise AIRemixError(
            "ACE-Step API 응답 형식이 올바르지 않습니다."
        )

    code = payload.get(
        "code",
        200,
    )

    if (
        code is not None
        and int(
            code
        )
        != 200
    ):
        raise AIRemixError(
            str(
                payload.get(
                    "error"
                )
                or payload
            )
        )

    error = payload.get(
        "error"
    )

    if error:
        raise AIRemixError(
            str(
                error
            )
        )

    return payload.get(
        "data"
    )


_DIRECT_UPLOAD_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
}

_FFMPEG_COVER_EXTENSIONS = {
    ".m4a",
    ".mp4",
    ".aac",
    ".webm",
    ".mkv",
    ".mov",
    ".wma",
    ".m4v",
}


def _prepare_cover_upload_audio(
    source_path: Path,
    *,
    log_callback: LogCallback | None = None,
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    ACE-Step accepts uploaded local source audio through multipart/form-data.

    WAV/MP3/FLAC/OGG/OPUS are uploaded directly.
    Video/container formats such as MP4/M4A are first extracted to a
    44.1 kHz stereo PCM WAV so the music-generation backend receives a
    conventional audio file without losing stereo information.
    """
    source = Path(
        source_path
    ).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    suffix = source.suffix.lower()

    if suffix in _DIRECT_UPLOAD_EXTENSIONS:
        return (
            source,
            None,
        )

    ffmpeg = find_ffmpeg()

    if not ffmpeg:
        raise AIRemixError(
            "AI Remix 원곡을 WAV로 준비하려면 FFmpeg가 필요합니다.\n"
            f"입력 형식: {suffix or '(확장자 없음)'}"
        )

    temp_dir = tempfile.TemporaryDirectory(
        prefix="ace_step_cover_"
    )
    output = (
        Path(
            temp_dir.name
        )
        / "cover_source.wav"
    )

    if log_callback is not None:
        log_callback(
            (
                "[ACE-Step] Source 변환: "
                f"{suffix or 'unknown'} → 44.1kHz stereo WAV"
            )
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(
            source
        ),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(
            output
        ),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        ),
        check=False,
    )

    if (
        completed.returncode != 0
        or not output.is_file()
        or output.stat().st_size <= 0
    ):
        error_text = (
            completed.stderr
            or completed.stdout
            or ""
        ).strip()

        temp_dir.cleanup()

        raise AIRemixError(
            "ACE-Step용 원곡 WAV 추출에 실패했습니다.\n\n"
            + error_text[
                -5000:
            ]
        )

    return (
        output,
        temp_dir,
    )


def _multipart_field_bytes(
    boundary: str,
    name: str,
    value: object,
) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n'
        "\r\n"
        f"{value}\r\n"
    ).encode(
        "utf-8"
    )


def _multipart_file_header(
    boundary: str,
    field_name: str,
    file_path: Path,
) -> bytes:
    content_type = (
        mimetypes.guess_type(
            file_path.name
        )[
            0
        ]
        or "application/octet-stream"
    )

    # Use a simple ASCII upload filename. The original local filename does
    # not need to be exposed to ACE-Step and can contain Korean/Japanese
    # characters that complicate multipart header compatibility.
    upload_name = (
        "source"
        + (
            file_path.suffix.lower()
            or ".wav"
        )
    )

    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{upload_name}"\r\n'
        f"Content-Type: {content_type}\r\n"
        "\r\n"
    ).encode(
        "utf-8"
    )


def _post_multipart_audio(
    *,
    host: str,
    port: int,
    endpoint: str,
    fields: dict[str, object],
    file_field: str,
    file_path: Path,
    timeout: float = 180.0,
) -> dict:
    """
    Stream multipart upload with http.client.

    This avoids loading a several-minute WAV entirely into RAM and matches
    ACE-Step's documented local-file upload path (`src_audio`).
    """
    boundary = (
        "----VocalPitchACE"
        + secrets.token_hex(
            16
        )
    )

    field_parts = [
        _multipart_field_bytes(
            boundary,
            name,
            value,
        )
        for name, value
        in fields.items()
    ]

    file_header = _multipart_file_header(
        boundary,
        file_field,
        file_path,
    )

    closing = (
        f"\r\n--{boundary}--\r\n"
    ).encode(
        "utf-8"
    )

    content_length = (
        sum(
            len(
                part
            )
            for part in field_parts
        )
        + len(
            file_header
        )
        + int(
            file_path.stat().st_size
        )
        + len(
            closing
        )
    )

    connection = http.client.HTTPConnection(
        host,
        int(
            port
        ),
        timeout=timeout,
    )

    try:
        connection.putrequest(
            "POST",
            endpoint,
        )
        connection.putheader(
            "Content-Type",
            (
                "multipart/form-data; "
                f"boundary={boundary}"
            ),
        )
        connection.putheader(
            "Content-Length",
            str(
                content_length
            ),
        )
        connection.putheader(
            "Accept",
            "application/json",
        )
        connection.endheaders()

        for part in field_parts:
            connection.send(
                part
            )

        connection.send(
            file_header
        )

        with file_path.open(
            "rb"
        ) as handle:
            while True:
                chunk = handle.read(
                    1024
                    * 1024
                )

                if not chunk:
                    break

                connection.send(
                    chunk
                )

        connection.send(
            closing
        )

        response = connection.getresponse()
        raw = response.read()

        text = raw.decode(
            "utf-8",
            errors="replace",
        )

        if not (
            200
            <= response.status
            < 300
        ):
            raise AIRemixError(
                (
                    "ACE-Step API HTTP "
                    f"{response.status}: "
                    f"{text}"
                )
            )

        try:
            payload = json.loads(
                text
            )
        except Exception as exc:
            raise AIRemixError(
                "ACE-Step API가 multipart 업로드 후 "
                "JSON이 아닌 응답을 반환했습니다."
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise AIRemixError(
                "ACE-Step API multipart 응답 형식이 올바르지 않습니다."
            )

        return payload

    except OSError as exc:
        raise AIRemixError(
            "ACE-Step source audio 업로드 연결 실패: "
            f"{exc}"
        ) from exc

    finally:
        connection.close()


def submit_cover_task(
    source_path: Path,
    *,
    prompt: str,
    lyrics: str,
    cover_strength: float,
    random_seed: bool,
    seed: int,
    vocal_language: str,
    host: str,
    port: int,
    log_callback: LogCallback | None = None,
) -> str:
    """
    Submit Cover using ACE-Step's documented multipart `src_audio` upload.

    Do not send a Windows absolute path as JSON `src_audio_path`.
    Current ACE-Step rejects arbitrary absolute paths for security.
    """
    prepared_audio = None
    temp_dir = None

    try:
        prepared_audio, temp_dir = (
            _prepare_cover_upload_audio(
                Path(
                    source_path
                ),
                log_callback=log_callback,
            )
        )

        if log_callback is not None:
            log_callback(
                (
                    "[ACE-Step] Source audio multipart 업로드: "
                    f"{prepared_audio.name} "
                    f"({prepared_audio.stat().st_size / (1024 * 1024):.1f} MiB)"
                )
            )

        fields: dict[str, object] = {
            "task_type": "cover",
            "prompt": str(
                prompt
            ),
            "lyrics": str(
                lyrics
                or ""
            ),
            "vocal_language": str(
                vocal_language
                or "ko"
            ),
            "audio_format": "wav",
            "model": DEFAULT_DIT_MODEL,
            "inference_steps": 8,
            "batch_size": 1,
            "audio_cover_strength": (
                f"{float(max(0.0, min(1.0, cover_strength))):.6f}"
            ),
            "thinking": "false",
            "use_random_seed": (
                "true"
                if random_seed
                else "false"
            ),
            "seed": (
                "-1"
                if random_seed
                else str(
                    int(
                        seed
                    )
                )
            ),
            "use_cot_caption": "false",
            "use_cot_language": "false",
        }

        response = _post_multipart_audio(
            host=host,
            port=port,
            endpoint="/release_task",
            fields=fields,
            file_field="src_audio",
            file_path=prepared_audio,
            timeout=240.0,
        )

        data = _unwrap_response(
            response
        )

        if not isinstance(
            data,
            dict,
        ):
            raise AIRemixError(
                "ACE-Step 작업 생성 응답에 data 객체가 없습니다."
            )

        task_id = str(
            data.get(
                "task_id",
                "",
            )
        ).strip()

        if not task_id:
            raise AIRemixError(
                "ACE-Step 작업 ID를 받지 못했습니다."
            )

        return task_id

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def query_task(
    task_id: str,
    *,
    host: str,
    port: int,
) -> dict:
    response = _http_json(
        "POST",
        api_base_url(
            host,
            port,
        )
        + "/query_result",
        {
            "task_id_list": [
                task_id
            ],
        },
        timeout=30.0,
    )

    data = _unwrap_response(
        response
    )

    if not isinstance(
        data,
        list,
    ) or not data:
        raise AIRemixError(
            "ACE-Step 작업 조회 결과가 비어 있습니다."
        )

    item = data[
        0
    ]

    if not isinstance(
        item,
        dict,
    ):
        raise AIRemixError(
            "ACE-Step 작업 조회 형식이 올바르지 않습니다."
        )

    return item


def _parse_result_items(
    item: dict,
) -> list[dict]:
    raw = item.get(
        "result"
    )

    if raw is None:
        return []

    if isinstance(
        raw,
        list,
    ):
        return [
            value
            for value in raw
            if isinstance(
                value,
                dict,
            )
        ]

    if isinstance(
        raw,
        dict,
    ):
        return [
            raw
        ]

    if isinstance(
        raw,
        str,
    ):
        text = raw.strip()

        if not text:
            return []

        try:
            parsed = json.loads(
                text
            )
        except Exception:
            return []

        if isinstance(
            parsed,
            list,
        ):
            return [
                value
                for value in parsed
                if isinstance(
                    value,
                    dict,
                )
            ]

        if isinstance(
            parsed,
            dict,
        ):
            return [
                parsed
            ]

    return []


def _download_result(
    result_item: dict,
    destination: Path,
    *,
    host: str,
    port: int,
) -> None:
    file_value = str(
        result_item.get(
            "file",
            "",
        )
        or result_item.get(
            "url",
            "",
        )
    ).strip()

    if not file_value:
        raise AIRemixError(
            "ACE-Step 결과에 다운로드 파일 경로가 없습니다."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_candidate = Path(
        file_value
    )

    if (
        local_candidate.is_absolute()
        and local_candidate.is_file()
    ):
        shutil.copy2(
            local_candidate,
            destination,
        )
        return

    if file_value.startswith(
        "http://"
    ) or file_value.startswith(
        "https://"
    ):
        download_url = (
            file_value
        )

    elif file_value.startswith(
        "/"
    ):
        download_url = (
            api_base_url(
                host,
                port,
            )
            + file_value
        )

    else:
        download_url = (
            api_base_url(
                host,
                port,
            )
            + "/"
            + file_value
        )

    try:
        with urlrequest.urlopen(
            download_url,
            timeout=180.0,
        ) as response:
            with destination.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(
                    response,
                    handle,
                )
    except Exception as exc:
        raise AIRemixError(
            "ACE-Step 생성 파일 다운로드 실패: "
            f"{exc}"
        ) from exc

    if (
        not destination.is_file()
        or destination.stat().st_size
        <= 0
    ):
        raise AIRemixError(
            "ACE-Step 결과 파일이 비어 있습니다."
        )


def generate_ai_remix(
    source_path: str | Path,
    output_path: str | Path,
    *,
    style_key: str,
    custom_prompt: str = "",
    lyrics: str = "",
    cover_strength: float = 0.45,
    random_seed: bool = True,
    seed: int = 12345,
    vocal_language: str = "ko",
    host: str = DEFAULT_API_HOST,
    port: int = DEFAULT_API_PORT,
    progress: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> AIRemixResult:
    source = Path(
        source_path
    ).expanduser().resolve()

    output = Path(
        output_path
    ).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    prompt = style_prompt(
        style_key,
        custom_prompt,
    )

    if not prompt:
        raise AIRemixError(
            "AI 리믹스 스타일 설명이 비어 있습니다."
        )

    lines: list[str] = []

    _emit(
        lines,
        log_callback,
        "ACE-Step 1.5 AI Remix 시작",
    )
    _emit(
        lines,
        log_callback,
        f"Source: {source}",
    )
    _emit(
        lines,
        log_callback,
        f"Style: {style_key}",
    )
    _emit(
        lines,
        log_callback,
        f"Prompt: {prompt}",
    )
    _emit(
        lines,
        log_callback,
        (
            "Cover strength: "
            f"{float(cover_strength):.2f}"
        ),
    )

    _progress(
        progress,
        1,
        "ACE-Step 환경 확인 중...",
    )

    ensure_api_server(
        host=host,
        port=port,
        log_callback=lambda text: (
            _emit(
                lines,
                log_callback,
                text,
            )
        ),
        progress=progress,
        cancel_check=cancel_check,
    )

    if (
        cancel_check is not None
        and cancel_check()
    ):
        raise AIRemixCancelled(
            "AI 리믹스가 중지되었습니다."
        )

    _progress(
        progress,
        18,
        "AI Remix 작업 제출 중...",
    )

    task_id = submit_cover_task(
        source,
        prompt=prompt,
        lyrics=lyrics,
        cover_strength=cover_strength,
        random_seed=random_seed,
        seed=seed,
        vocal_language=vocal_language,
        host=host,
        port=port,
        log_callback=lambda text: (
            _emit(
                lines,
                log_callback,
                text,
            )
        ),
    )

    _emit(
        lines,
        log_callback,
        f"Task ID: {task_id}",
    )

    _progress(
        progress,
        22,
        "ACE-Step이 재편곡을 생성하는 중...",
    )

    started = time.monotonic()
    last_log_second = -1
    result_item: dict | None = None

    while True:
        if (
            cancel_check is not None
            and cancel_check()
        ):
            raise AIRemixCancelled(
                "GUI에서 대기를 중지했습니다. "
                "이미 제출된 ACE-Step 작업은 서버에서 계속 진행될 수 있습니다."
            )

        status_item = query_task(
            task_id,
            host=host,
            port=port,
        )

        status = int(
            status_item.get(
                "status",
                0,
            )
            or 0
        )

        if status == 1:
            items = _parse_result_items(
                status_item
            )

            if not items:
                raise AIRemixError(
                    "ACE-Step은 성공 상태를 반환했지만 결과 오디오가 없습니다."
                )

            result_item = items[
                0
            ]
            break

        if status == 2:
            raise AIRemixError(
                "ACE-Step AI Remix 생성이 실패했습니다.\n\n"
                + str(
                    status_item.get(
                        "error"
                    )
                    or status_item.get(
                        "result"
                    )
                    or status_item
                )
            )

        elapsed = int(
            time.monotonic()
            - started
        )

        if (
            elapsed
            // 10
            != last_log_second
        ):
            last_log_second = (
                elapsed
                // 10
            )

            _emit(
                lines,
                log_callback,
                (
                    "[ACE-Step] 생성 중... "
                    f"{elapsed}s"
                ),
            )

        # Unknown true percentage: slowly move UI from 22 to 88.
        visual = min(
            88,
            22
            + int(
                elapsed
                / 6
            ),
        )

        _progress(
            progress,
            visual,
            (
                "ACE-Step AI 재편곡 생성 중... "
                f"{elapsed}s"
            ),
        )

        time.sleep(
            2.0
        )

    assert result_item is not None

    _progress(
        progress,
        92,
        "AI Remix 결과 다운로드 중...",
    )

    _download_result(
        result_item,
        output,
        host=host,
        port=port,
    )

    _progress(
        progress,
        100,
        f"AI Remix 완료: {output.name}",
    )

    metadata = {
        "status": "success",
        "source_path": str(
            source
        ),
        "output_path": str(
            output
        ),
        "task_id": task_id,
        "style_key": style_key,
        "prompt": prompt,
        "lyrics_provided": bool(
            str(
                lyrics
                or ""
            ).strip()
        ),
        "cover_strength": float(
            cover_strength
        ),
        "random_seed": bool(
            random_seed
        ),
        "seed": (
            None
            if random_seed
            else int(
                seed
            )
        ),
        "result": result_item,
        "engine": "ACE-Step 1.5 cover",
        "dit_model": DEFAULT_DIT_MODEL,
    }

    _write_logs(
        lines,
        metadata,
    )

    return AIRemixResult(
        source_path=source,
        output_path=output,
        task_id=task_id,
        style_key=style_key,
        prompt=prompt,
        cover_strength=float(
            cover_strength
        ),
        seed=(
            None
            if random_seed
            else int(
                seed
            )
        ),
        metadata=metadata,
    )
