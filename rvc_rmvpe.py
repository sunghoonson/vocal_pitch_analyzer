from __future__ import annotations

from pathlib import Path
from typing import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
import re
import subprocess
import tempfile
import traceback

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
from rvc_lead_selector import (
    LeadVocalSelectorError,
    select_lead_vocal,
)

# V25_RVC_ARTIFACT_GUARD_PATCH
# V27_LEAD_VOCAL_SELECTOR_PATCH
# V26_ARTIFACT_PRIORITY_MANUAL_BYPASS_PATCH
# V25_ADAPTIVE_GUARD_DIAGNOSTICS_HOTFIX
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



def adaptive_guard_log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_adaptive_guard_last.log"
    )


def adaptive_guard_json_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_adaptive_guard_last.json"
    )


def _guard_timestamp() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def _guard_safe_path(
    value: str | Path | None,
) -> str | None:
    if value is None:
        return None

    try:
        return str(
            Path(
                value
            ).resolve()
        )
    except Exception:
        return str(
            value
        )


def _guard_write_json(
    state: dict,
) -> None:
    path = adaptive_guard_json_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state[
            "updated_at"
        ] = _guard_timestamp()

        temp_path = path.with_suffix(
            path.suffix
            + ".tmp"
        )

        temp_path.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        temp_path.replace(
            path
        )
    except Exception:
        # Diagnostic output must never break RVC conversion.
        pass


def _guard_append_log(
    message: str,
) -> None:
    clean = str(
        message
    ).strip()

    if not clean:
        return

    path = adaptive_guard_log_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"[{_guard_timestamp()}] {clean}\n"
            )
    except OSError:
        pass


def _guard_reset_log() -> None:
    path = adaptive_guard_log_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    except OSError:
        pass


def _guard_emit(
    callback: LogCallback | None,
    message: str,
) -> None:
    _guard_append_log(
        message
    )
    _emit(
        callback,
        message,
    )


def _guard_start_state(
    *,
    source_vocal: Path,
    raw_rvc: Path,
    model_path: str | Path,
    index_path: str | Path | None,
    semitones: int,
    harmony_guard_enabled: bool,
    harmony_guard_sensitivity: str,
    harmony_guard_crossfade_ms: int,
    manual_bypass_ranges: list[tuple[float, float]] | None = None,
) -> dict:
    state = {
        "guard_version": "2.5",
        "diagnostics_hotfix": True,
        "status": "started",
        "stage": "prepare",
        "started_at": _guard_timestamp(),
        "updated_at": _guard_timestamp(),
        "message": (
            "Adaptive Guard diagnostic session started."
        ),
        "fallback_to_raw_rvc": False,
        "source_vocal": _guard_safe_path(
            source_vocal
        ),
        "raw_rvc": _guard_safe_path(
            raw_rvc
        ),
        "model_path": _guard_safe_path(
            model_path
        ),
        "index_path": _guard_safe_path(
            index_path
        ),
        "semitones": int(
            semitones
        ),
        "enabled": bool(
            harmony_guard_enabled
        ),
        "sensitivity": str(
            harmony_guard_sensitivity
        ),
        "crossfade_ms": int(
            harmony_guard_crossfade_ms
        ),
        "manual_bypass_ranges": [
            [
                float(start),
                float(end),
            ]
            for start, end in (
                manual_bypass_ranges
                or []
            )
        ],
        "manual_region_count": len(
            manual_bypass_ranges
            or []
        ),
        "pitch_only": None,
        "adaptive_output": None,
        "rubberband_available": None,
        "rubberband_status": None,
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
    }

    _guard_write_json(
        state
    )

    return state


def _guard_update_state(
    state: dict,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    fallback_to_raw_rvc: bool | None = None,
    exception: BaseException | None = None,
    extra: dict | None = None,
) -> None:
    if status is not None:
        state[
            "status"
        ] = str(
            status
        )

    if stage is not None:
        state[
            "stage"
        ] = str(
            stage
        )

    if message is not None:
        state[
            "message"
        ] = str(
            message
        )

    if fallback_to_raw_rvc is not None:
        state[
            "fallback_to_raw_rvc"
        ] = bool(
            fallback_to_raw_rvc
        )

    if exception is not None:
        state[
            "exception_type"
        ] = type(
            exception
        ).__name__
        state[
            "exception_message"
        ] = str(
            exception
        )
        state[
            "traceback"
        ] = "".join(
            traceback.format_exception(
                type(
                    exception
                ),
                exception,
                exception.__traceback__,
            )
        )

    if extra:
        state.update(
            extra
        )

    _guard_write_json(
        state
    )


def _guard_report_dict(
    report,
) -> dict:
    try:
        if is_dataclass(
            report
        ):
            payload = asdict(
                report
            )
        elif hasattr(
            report,
            "__dict__",
        ):
            payload = dict(
                report.__dict__
            )
        else:
            payload = {
                "report": str(
                    report
                )
            }
    except Exception as exc:
        payload = {
            "report_serialization_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "report": str(
                report
            ),
        }

    return payload



def _parse_time_token(
    token: str,
) -> float:
    value = str(token).strip()
    if not value:
        raise ValueError("빈 시간 값입니다.")

    parts = value.split(":")

    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            seconds = (
                float(parts[0])
                * 60.0
                + float(parts[1])
            )
        elif len(parts) == 3:
            seconds = (
                float(parts[0])
                * 3600.0
                + float(parts[1])
                * 60.0
                + float(parts[2])
            )
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"시간 형식을 읽을 수 없습니다: {value}"
        ) from exc

    if seconds < 0.0:
        raise ValueError(
            f"시간은 0 이상이어야 합니다: {value}"
        )

    return float(seconds)


def parse_manual_bypass_ranges(
    text: str,
) -> list[tuple[float, float]]:
    source = str(text or "").strip()

    if not source:
        return []

    chunks = [
        chunk.strip()
        for chunk in re.split(
            r"[\n,;]+",
            source,
        )
        if chunk.strip()
    ]

    result: list[tuple[float, float]] = []

    for chunk in chunks:
        match = re.match(
            r"^\s*(.+?)\s*(?:-|~|–|—|→)\s*(.+?)\s*$",
            chunk,
        )
        if not match:
            raise ValueError(
                "수동 우회 구간 형식 오류: "
                f"'{chunk}'\n"
                "예: 00:42.300 - 00:44.100"
            )

        start = _parse_time_token(
            match.group(1)
        )
        end = _parse_time_token(
            match.group(2)
        )

        if end <= start:
            raise ValueError(
                "수동 우회 종료 시간은 시작 시간보다 커야 합니다: "
                f"'{chunk}'"
            )

        result.append((start, end))

    result.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    merged: list[tuple[float, float]] = []

    for start, end in result:
        if (
            merged
            and start
            <= merged[-1][1]
            + 0.001
        ):
            merged[-1] = (
                merged[-1][0],
                max(
                    merged[-1][1],
                    end,
                ),
            )
        else:
            merged.append((start, end))

    return merged


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
    manual_bypass_ranges: list[tuple[float, float]] | None,
    progress: ProgressCallback | None,
    progress_points: tuple[int, int, int, int],
    log_callback: LogCallback | None,
) -> Path:
    (
        start_percent,
        rvc_percent,
        fallback_percent,
        blend_percent,
    ) = progress_points

    raw_rvc = (
        temp_dir
        / "rvc_vocal_raw.wav"
    )

    # Start diagnostics BEFORE RVC inference so a file is guaranteed
    # even if the Guard is never reached.
    _guard_reset_log()

    guard_state = _guard_start_state(
        source_vocal=source_vocal,
        raw_rvc=raw_rvc,
        model_path=model_path,
        index_path=index_path,
        semitones=semitones,
        harmony_guard_enabled=harmony_guard_enabled,
        harmony_guard_sensitivity=harmony_guard_sensitivity,
        harmony_guard_crossfade_ms=harmony_guard_crossfade_ms,
        manual_bypass_ranges=manual_bypass_ranges,
    )

    _guard_emit(
        log_callback,
        (
            "[Adaptive Guard Diagnostics] 시작 "
            f"(enabled={bool(harmony_guard_enabled)}, "
            f"sensitivity={harmony_guard_sensitivity}, "
            f"crossfade={int(harmony_guard_crossfade_ms)}ms)"
        ),
    )
    _guard_emit(
        log_callback,
        (
            "[Adaptive Guard Diagnostics] LOG: "
            f"{adaptive_guard_log_path()}"
        ),
    )
    _guard_emit(
        log_callback,
        (
            "[Adaptive Guard Diagnostics] JSON: "
            f"{adaptive_guard_json_path()}"
        ),
    )

    _progress(
        progress,
        start_percent,
        "RMVPE F0 추출 + RVC 음색 변환 중...",
    )

    _guard_update_state(
        guard_state,
        status="running",
        stage="rvc_inference",
        message="RVC + RMVPE inference is running.",
    )

    try:
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
    except Exception as exc:
        message = (
            "[Adaptive Guard Diagnostics] RVC 추론 단계에서 실패했습니다: "
            f"{type(exc).__name__}: {exc}"
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="failed",
            stage="rvc_inference",
            message=message,
            fallback_to_raw_rvc=False,
            exception=exc,
        )
        raise

    _guard_update_state(
        guard_state,
        status="running",
        stage="rvc_complete",
        message="Raw RVC vocal was created successfully.",
        extra={
            "raw_rvc_exists": bool(
                raw_rvc.is_file()
            ),
            "raw_rvc_size": (
                int(
                    raw_rvc.stat().st_size
                )
                if raw_rvc.is_file()
                else 0
            ),
        },
    )

    _progress(
        progress,
        rvc_percent,
        "RVC 보컬 생성 완료.",
    )

    manual_ranges = list(
        manual_bypass_ranges
        or []
    )

    if (
        not harmony_guard_enabled
        and not manual_ranges
    ):
        message = (
            "[Adaptive Guard] 자동 Guard OFF / 수동 우회 없음 - "
            "기존 RVC 출력 그대로 사용"
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="disabled",
            stage="settings",
            message=message,
            fallback_to_raw_rvc=True,
        )
        return raw_rvc

    if (
        not harmony_guard_enabled
        and manual_ranges
    ):
        _guard_emit(
            log_callback,
            (
                "[Manual Bypass] 자동 Guard는 OFF지만 "
                f"수동 우회 {len(manual_ranges)}개 구간이 있어 "
                "Pitch-only 경로를 생성합니다."
            ),
        )

    _guard_update_state(
        guard_state,
        status="running",
        stage="rubberband_check",
        message="Checking RubberBand for Pitch-only fallback.",
    )

    try:
        rb_ok, rb_status = (
            rubberband_filter_available()
        )
    except Exception as exc:
        message = (
            "[Adaptive Guard] RubberBand 확인 중 예외 - "
            "기존 RVC 출력으로 계속합니다: "
            f"{type(exc).__name__}: {exc}"
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="failed",
            stage="rubberband_check",
            message=message,
            fallback_to_raw_rvc=True,
            exception=exc,
        )
        return raw_rvc

    _guard_update_state(
        guard_state,
        extra={
            "rubberband_available": bool(
                rb_ok
            ),
            "rubberband_status": str(
                rb_status
            ),
        },
    )

    if not rb_ok:
        message = (
            "[Adaptive Guard] Pitch-only 우회용 RubberBand를 사용할 수 없어 "
            "Guard를 건너뜁니다: "
            + str(
                rb_status
            )
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="skipped",
            stage="rubberband_check",
            message=message,
            fallback_to_raw_rvc=True,
        )
        return raw_rvc

    pitch_only = (
        temp_dir
        / "pitch_only_vocal.wav"
    )

    guard_state[
        "pitch_only"
    ] = _guard_safe_path(
        pitch_only
    )

    _guard_update_state(
        guard_state,
        status="running",
        stage="pitch_only_generation",
        message="Generating same-key Pitch-only fallback vocal.",
    )

    _progress(
        progress,
        min(
            fallback_percent,
            99,
        ),
        "Adaptive Guard: Pitch-only 우회 보컬 생성 중...",
    )

    try:
        transpose_audio(
            source_vocal,
            pitch_only,
            semitones=semitones,
            preserve_formant=True,
            quality="quality",
        )
    except Exception as exc:
        message = (
            "[Adaptive Guard] Pitch-only 보컬 생성 실패 - "
            "기존 RVC 출력으로 계속합니다: "
            f"{type(exc).__name__}: {exc}"
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="failed",
            stage="pitch_only_generation",
            message=message,
            fallback_to_raw_rvc=True,
            exception=exc,
            extra={
                "pitch_only_exists": bool(
                    pitch_only.is_file()
                ),
                "pitch_only_size": (
                    int(
                        pitch_only.stat().st_size
                    )
                    if pitch_only.is_file()
                    else 0
                ),
            },
        )
        return raw_rvc

    _guard_update_state(
        guard_state,
        status="running",
        stage="pitch_only_complete",
        message="Pitch-only fallback vocal created.",
        extra={
            "pitch_only_exists": bool(
                pitch_only.is_file()
            ),
            "pitch_only_size": (
                int(
                    pitch_only.stat().st_size
                )
                if pitch_only.is_file()
                else 0
            ),
        },
    )

    adaptive = (
        temp_dir
        / "rvc_vocal_harmony_guard.wav"
    )

    guard_state[
        "adaptive_output"
    ] = _guard_safe_path(
        adaptive
    )

    _guard_update_state(
        guard_state,
        status="running",
        stage="harmony_artifact_analysis",
        message=(
            "Running input Harmony analysis and RVC Artifact analysis."
        ),
    )

    _progress(
        progress,
        min(
            blend_percent,
            99,
        ),
        (
            "Adaptive Guard v2.5: 입력 Harmony + "
            "RVC 출력 Artifact 2차 검증 중..."
        ),
    )

    def guard_callback(
        message: str,
    ) -> None:
        _guard_emit(
            log_callback,
            message,
        )

    try:
        report = blend_adaptive_vocals(
            source_vocal,
            raw_rvc,
            pitch_only,
            adaptive,
            sensitivity=harmony_guard_sensitivity,
            crossfade_ms=harmony_guard_crossfade_ms,
            auto_guard_enabled=bool(
                harmony_guard_enabled
            ),
            manual_bypass_ranges=manual_ranges,
            log_callback=guard_callback,
        )
    except Exception as exc:
        message = (
            "[Adaptive Guard] 분석/Blend 실패 - "
            "기존 RVC 출력으로 계속합니다: "
            f"{type(exc).__name__}: {exc}"
        )
        _guard_emit(
            log_callback,
            message,
        )
        _guard_update_state(
            guard_state,
            status="failed",
            stage="harmony_artifact_analysis",
            message=message,
            fallback_to_raw_rvc=True,
            exception=exc,
            extra={
                "adaptive_output_exists": bool(
                    adaptive.is_file()
                ),
                "adaptive_output_size": (
                    int(
                        adaptive.stat().st_size
                    )
                    if adaptive.is_file()
                    else 0
                ),
            },
        )
        return raw_rvc

    report_dict = (
        _guard_report_dict(
            report
        )
    )

    message = (
        "[Adaptive Guard v2.6] 적용 완료: "
        f"총 위험 구간 {report.risky_region_count}개, "
        f"Artifact 구간 {report.artifact_region_count}개, "
        f"Pitch-only 우회 {report.fallback_seconds:.1f}s, "
        f"부분 Blend {report.blend_seconds:.1f}s"
    )

    _guard_emit(
        log_callback,
        message,
    )

    # Keep report fields at top-level for compatibility with the
    # previous rvc_adaptive_guard_last.json format.
    success_extra = dict(
        report_dict
    )
    success_extra.update(
        {
            "adaptive_output_exists": bool(
                adaptive.is_file()
            ),
            "adaptive_output_size": (
                int(
                    adaptive.stat().st_size
                )
                if adaptive.is_file()
                else 0
            ),
            "report": report_dict,
        }
    )

    _guard_update_state(
        guard_state,
        status="success",
        stage="complete",
        message=message,
        fallback_to_raw_rvc=False,
        extra=success_extra,
    )

    _guard_emit(
        log_callback,
        (
            "[Adaptive Guard Diagnostics] 완료. "
            f"JSON={adaptive_guard_json_path()}"
        ),
    )

    return adaptive

def _prepare_rvc_vocal_pipeline(
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
    lead_selector_enabled: bool,
    lead_selector_strength: str,
    harmony_guard_enabled: bool,
    harmony_guard_sensitivity: str,
    harmony_guard_crossfade_ms: int,
    manual_bypass_ranges: list[tuple[float, float]] | None,
    progress: ProgressCallback | None,
    selector_progress: tuple[int, int, int],
    adaptive_progress: tuple[int, int, int, int],
    log_callback: LogCallback | None,
) -> Path:
    (
        selector_start,
        selector_done,
        residual_shift_percent,
    ) = selector_progress

    if not lead_selector_enabled:
        _emit(
            log_callback,
            (
                "[Lead Selector] OFF - 기존 전체 vocal stem을 "
                "그대로 RVC에 전달합니다."
            ),
        )

        return _prepare_adaptive_rvc_vocal(
            source_vocal,
            temp_dir,
            model_path=model_path,
            index_path=index_path,
            semitones=semitones,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            harmony_guard_enabled=harmony_guard_enabled,
            harmony_guard_sensitivity=harmony_guard_sensitivity,
            harmony_guard_crossfade_ms=harmony_guard_crossfade_ms,
            manual_bypass_ranges=manual_bypass_ranges,
            progress=progress,
            progress_points=adaptive_progress,
            log_callback=log_callback,
        )

    lead_candidate = (
        temp_dir
        / "lead_candidate.wav"
    )
    nonlead_residual = (
        temp_dir
        / "nonlead_residual.wav"
    )

    _progress(
        progress,
        selector_start,
        "Lead Vocal Selector: 메인 보컬 선별 중...",
    )

    try:
        selector_report = select_lead_vocal(
            source_vocal,
            lead_candidate,
            nonlead_residual,
            strength=lead_selector_strength,
            log_callback=log_callback,
            save_debug_copy=True,
        )
    except Exception as exc:
        _emit(
            log_callback,
            (
                "[Lead Selector] 실패 - 안전하게 기존 전체 vocal "
                "RVC 경로로 되돌립니다: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

        return _prepare_adaptive_rvc_vocal(
            source_vocal,
            temp_dir,
            model_path=model_path,
            index_path=index_path,
            semitones=semitones,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            harmony_guard_enabled=harmony_guard_enabled,
            harmony_guard_sensitivity=harmony_guard_sensitivity,
            harmony_guard_crossfade_ms=harmony_guard_crossfade_ms,
            manual_bypass_ranges=manual_bypass_ranges,
            progress=progress,
            progress_points=adaptive_progress,
            log_callback=log_callback,
        )

    minimum_seconds = max(
        1.0,
        selector_report.duration_seconds
        * 0.025,
    )

    if (
        selector_report.selected_seconds
        < minimum_seconds
        or selector_report.lead_energy_ratio
        < 0.008
    ):
        _emit(
            log_callback,
            (
                "[Lead Selector] 선택된 Lead가 너무 적어 "
                "전체 vocal RVC 경로로 되돌립니다. "
                f"(selected={selector_report.selected_seconds:.1f}s, "
                f"energy={selector_report.lead_energy_ratio * 100.0:.2f}%)"
            ),
        )

        return _prepare_adaptive_rvc_vocal(
            source_vocal,
            temp_dir,
            model_path=model_path,
            index_path=index_path,
            semitones=semitones,
            index_rate=index_rate,
            protect=protect,
            rms_mix_rate=rms_mix_rate,
            speaker_id=speaker_id,
            harmony_guard_enabled=harmony_guard_enabled,
            harmony_guard_sensitivity=harmony_guard_sensitivity,
            harmony_guard_crossfade_ms=harmony_guard_crossfade_ms,
            manual_bypass_ranges=manual_bypass_ranges,
            progress=progress,
            progress_points=adaptive_progress,
            log_callback=log_callback,
        )

    _progress(
        progress,
        selector_done,
        (
            "Lead Vocal Selector 완료: "
            f"Lead {selector_report.selected_ratio * 100.0:.1f}%"
        ),
    )

    converted_lead = _prepare_adaptive_rvc_vocal(
        lead_candidate,
        temp_dir,
        model_path=model_path,
        index_path=index_path,
        semitones=semitones,
        index_rate=index_rate,
        protect=protect,
        rms_mix_rate=rms_mix_rate,
        speaker_id=speaker_id,
        harmony_guard_enabled=harmony_guard_enabled,
        harmony_guard_sensitivity=harmony_guard_sensitivity,
        harmony_guard_crossfade_ms=harmony_guard_crossfade_ms,
        manual_bypass_ranges=manual_bypass_ranges,
        progress=progress,
        progress_points=adaptive_progress,
        log_callback=log_callback,
    )

    shifted_nonlead = (
        temp_dir
        / "nonlead_residual_shifted.wav"
    )

    _progress(
        progress,
        residual_shift_percent,
        (
            "Backing/Harmony residual을 RVC 없이 "
            "같은 키로 이동하는 중..."
        ),
    )

    try:
        transpose_audio(
            nonlead_residual,
            shifted_nonlead,
            semitones=semitones,
            preserve_formant=True,
            quality="quality",
        )
    except AudioTransposeError as exc:
        raise RVCRMVPEError(
            "Lead RVC는 생성됐지만 Non-lead/화음 residual "
            "Pitch Shift에 실패했습니다.\n\n"
            f"{exc}"
        ) from exc

    combined_vocal = (
        temp_dir
        / "lead_selected_rvc_vocal.wav"
    )

    def mix_log(
        text: str,
    ) -> None:
        _emit(
            log_callback,
            str(
                text
            ).replace(
                "Seed-VC 보컬 레벨 보정",
                "RVC Lead 보컬 레벨 보정",
            ),
        )

    _mix_stems(
        converted_lead,
        shifted_nonlead,
        lead_candidate,
        combined_vocal,
        log_callback=mix_log,
    )

    _emit(
        log_callback,
        (
            "[Lead Selector] 재합성 완료: "
            "Lead=RVC / Non-lead·Harmony=Pitch Shift only"
        ),
    )

    return combined_vocal


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
    lead_selector_enabled: bool = True,
    lead_selector_strength: str = "balanced",
    harmony_guard_enabled: bool = True,
    harmony_guard_sensitivity: str = "medium",
    harmony_guard_crossfade_ms: int = 500,
    manual_bypass_ranges: list[tuple[float, float]] | None = None,
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
            _prepare_rvc_vocal_pipeline(
                vocal,
                temp_dir,
                model_path=model_path,
                index_path=index_path,
                semitones=semitones,
                index_rate=index_rate,
                protect=protect,
                rms_mix_rate=rms_mix_rate,
                speaker_id=speaker_id,
                lead_selector_enabled=bool(
                    lead_selector_enabled
                ),
                lead_selector_strength=str(
                    lead_selector_strength
                ),
                harmony_guard_enabled=bool(
                    harmony_guard_enabled
                ),
                harmony_guard_sensitivity=str(
                    harmony_guard_sensitivity
                ),
                harmony_guard_crossfade_ms=int(
                    harmony_guard_crossfade_ms
                ),
                manual_bypass_ranges=list(
                    manual_bypass_ranges
                    or []
                ),
                progress=progress,
                selector_progress=(
                    10,
                    20,
                    84,
                ),
                adaptive_progress=(
                    24,
                    66,
                    72,
                    78,
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
    lead_selector_enabled: bool = True,
    lead_selector_strength: str = "balanced",
    harmony_guard_enabled: bool = True,
    harmony_guard_sensitivity: str = "medium",
    harmony_guard_crossfade_ms: int = 500,
    manual_bypass_ranges: list[tuple[float, float]] | None = None,
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
            _prepare_rvc_vocal_pipeline(
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
                lead_selector_enabled=bool(
                    lead_selector_enabled
                ),
                lead_selector_strength=str(
                    lead_selector_strength
                ),
                harmony_guard_enabled=bool(
                    harmony_guard_enabled
                ),
                harmony_guard_sensitivity=str(
                    harmony_guard_sensitivity
                ),
                harmony_guard_crossfade_ms=int(
                    harmony_guard_crossfade_ms
                ),
                manual_bypass_ranges=list(
                    manual_bypass_ranges
                    or []
                ),
                progress=progress,
                selector_progress=(
                    27,
                    36,
                    79,
                ),
                adaptive_progress=(
                    40,
                    66,
                    71,
                    76,
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
