from __future__ import annotations

# V33_NEURAL_LEAD_BACKING_RVC_PATCH
#
# Second-stage vocal separation specifically for RVC input:
#
#   BS-RoFormer vocals
#       ↓
#   audio-separator "karaoke" ensemble
#       ↓
#   Lead Vocals + Backing/Harmony
#
# Lead only is sent to RVC.
# Backing/Harmony is pitch-shifted without RVC and mixed back later.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from vocal_separator import (
    find_separator_executable,
    model_cache_dir,
    _ffmpeg_env,
    _prepare_separator_input,
)


DEFAULT_KARAOKE_PRESET = "karaoke"
DEFAULT_KARAOKE_FALLBACK_MODEL = (
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
)

LogCallback = Callable[[str], None]


class NeuralLeadBackingError(RuntimeError):
    pass


@dataclass(slots=True)
class NeuralLeadBackingResult:
    source_vocal: Path
    lead_path: Path
    backing_path: Path
    preset: str
    cache_hit: bool
    used_fallback_model: bool
    lead_rms_db: float
    backing_rms_db: float
    duration_seconds: float
    lead_energy_ratio: float


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parent


def cache_root() -> Path:
    return (
        project_root()
        / "cache"
        / "rvc_neural_lead_backing"
    )


def log_dir() -> Path:
    return (
        project_root()
        / "logs"
    )


def log_path() -> Path:
    return (
        log_dir()
        / "rvc_neural_lead_backing_last.log"
    )


def json_path() -> Path:
    return (
        log_dir()
        / "rvc_neural_lead_backing_last.json"
    )


def debug_lead_path() -> Path:
    return (
        cache_root()
        / "last_lead.wav"
    )


def debug_backing_path() -> Path:
    return (
        cache_root()
        / "last_backing.wav"
    )


def debug_source_path() -> Path:
    return (
        cache_root()
        / "last_input_vocals.wav"
    )


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


def _cache_key(
    source: Path,
    preset: str,
) -> str:
    stat = source.stat()

    payload = {
        "path": str(
            source.resolve()
        ),
        "size": int(
            stat.st_size
        ),
        "mtime_ns": int(
            stat.st_mtime_ns
        ),
        "preset": str(
            preset
        ),
        "schema": 3,
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[
        :24
    ]


def _audio_stats(
    path: Path,
) -> tuple[
    float,
    float,
]:
    """
    Return duration seconds and RMS dBFS-like value from a WAV/FLAC/etc
    readable by libsndfile.
    """
    square_sum = 0.0
    sample_count = 0
    frame_count = 0
    samplerate = 0

    with sf.SoundFile(
        str(
            path
        )
    ) as handle:
        samplerate = int(
            handle.samplerate
        )

        for block in handle.blocks(
            blocksize=65536,
            dtype="float32",
            always_2d=True,
        ):
            array = np.asarray(
                block,
                dtype=np.float64,
            )

            if array.size <= 0:
                continue

            square_sum += float(
                np.sum(
                    array
                    * array
                )
            )
            sample_count += int(
                array.size
            )
            frame_count += int(
                array.shape[
                    0
                ]
            )

    duration = (
        float(
            frame_count
        )
        / float(
            samplerate
        )
        if samplerate > 0
        else 0.0
    )

    if sample_count <= 0:
        return (
            duration,
            -120.0,
        )

    rms = math.sqrt(
        max(
            square_sum
            / sample_count,
            0.0,
        )
    )

    db = (
        20.0
        * math.log10(
            max(
                rms,
                1e-8,
            )
        )
    )

    return (
        duration,
        float(
            db
        ),
    )


def _energy_ratio(
    lead_db: float,
    backing_db: float,
) -> float:
    lead_power = (
        10.0
        ** (
            float(
                lead_db
            )
            / 10.0
        )
    )
    backing_power = (
        10.0
        ** (
            float(
                backing_db
            )
            / 10.0
        )
    )

    total = (
        lead_power
        + backing_power
    )

    if total <= 0.0:
        return 0.0

    return float(
        lead_power
        / total
    )


def _score_lead_name(
    name: str,
) -> int:
    value = name.lower()

    score = 0

    if "lead_vocals" in value:
        score += 100
    if "lead vocals" in value:
        score += 100
    if "lead_only" in value:
        score += 90
    if "with_lead_vocals" in value:
        score += 80
    if "(vocals)" in value:
        score += 65
    if "_vocals_" in value:
        score += 50
    if value.endswith(
        "_vocals.wav"
    ):
        score += 40

    if "backing" in value:
        score -= 100
    if "instrumental" in value:
        score -= 100
    if "karaoke" in value:
        score -= 40

    return score


def _score_backing_name(
    name: str,
) -> int:
    value = name.lower()

    score = 0

    if "backing_vocals" in value:
        score += 100
    if "backing vocals" in value:
        score += 100
    if "backing_only" in value:
        score += 90
    if "with_backing_vocals" in value:
        score += 80
    if "(instrumental)" in value:
        score += 65
    if "instrumental" in value:
        score += 55
    if "karaoke" in value:
        score += 35

    if "lead_vocals" in value:
        score -= 100
    if "lead vocals" in value:
        score -= 100
    if "lead_only" in value:
        score -= 90

    return score


def classify_output_files(
    output_dir: Path,
) -> tuple[
    Path | None,
    Path | None,
]:
    """
    audio-separator 0.47 can expose karaoke stems as either
    Lead/Backing labels or Vocals/Instrumental labels depending on the
    underlying model metadata. custom_output_names normally makes these
    explicit, but this detector keeps the integration robust.
    """
    preferred_lead = (
        output_dir
        / "lead_vocals.wav"
    )
    preferred_backing = (
        output_dir
        / "backing_vocals.wav"
    )

    if (
        preferred_lead.is_file()
        and preferred_backing.is_file()
    ):
        return (
            preferred_lead,
            preferred_backing,
        )

    files = [
        path
        for path in output_dir.rglob(
            "*.wav"
        )
        if (
            path.is_file()
            and path.stat().st_size
            > 1024
        )
    ]

    if not files:
        return (
            None,
            None,
        )

    lead_ranked = sorted(
        files,
        key=lambda p: (
            _score_lead_name(
                p.name
            ),
            p.stat().st_size,
        ),
        reverse=True,
    )
    backing_ranked = sorted(
        files,
        key=lambda p: (
            _score_backing_name(
                p.name
            ),
            p.stat().st_size,
        ),
        reverse=True,
    )

    lead = (
        lead_ranked[
            0
        ]
        if (
            lead_ranked
            and _score_lead_name(
                lead_ranked[
                    0
                ].name
            )
            > 0
        )
        else None
    )

    backing = (
        backing_ranked[
            0
        ]
        if (
            backing_ranked
            and _score_backing_name(
                backing_ranked[
                    0
                ].name
            )
            > 0
        )
        else None
    )

    if (
        lead is not None
        and backing is not None
        and lead.resolve()
        == backing.resolve()
    ):
        # If only one file got a useful score, leave the other unresolved.
        if (
            _score_lead_name(
                lead.name
            )
            >= _score_backing_name(
                backing.name
            )
        ):
            backing = None
        else:
            lead = None

    # Exactly two outputs is common for karaoke models. If one was
    # identified confidently, the remaining file is the complementary stem.
    if len(
        files
    ) == 2:
        if (
            lead is not None
            and backing is None
        ):
            backing = next(
                (
                    p
                    for p in files
                    if p.resolve()
                    != lead.resolve()
                ),
                None,
            )

        elif (
            backing is not None
            and lead is None
        ):
            lead = next(
                (
                    p
                    for p in files
                    if p.resolve()
                    != backing.resolve()
                ),
                None,
            )

    return (
        lead,
        backing,
    )


def _separator_command(
    *,
    exe: str,
    source: Path,
    output_dir: Path,
    preset: str | None,
    model_filename: str | None,
    use_autocast: bool,
) -> list[str]:
    custom_names = json.dumps(
        {
            # Native Lead/Backing naming used by current audio-separator.
            "Lead Vocals": "lead_vocals",
            "Backing Vocals": "backing_vocals",

            # Karaoke RoFormers may surface their complementary pair as
            # Vocals / Instrumental. For a SECOND PASS over an already
            # isolated vocal stem, these mean Lead / Backing respectively.
            "Vocals": "lead_vocals",
            "Instrumental": "backing_vocals",

            # Additional compatibility labels.
            "Primary Stem": "lead_vocals",
            "Secondary Stem": "backing_vocals",
        },
        ensure_ascii=False,
    )

    command = [
        exe,
        str(
            source
        ),
        "--output_format",
        "WAV",
        "--output_dir",
        str(
            output_dir
        ),
        "--model_file_dir",
        str(
            model_cache_dir()
        ),
        "--sample_rate",
        "44100",
        "--custom_output_names",
        custom_names,
        "--log_level",
        "info",
    ]

    if preset:
        command.extend(
            [
                "--ensemble_preset",
                str(
                    preset
                ),
            ]
        )
    elif model_filename:
        command.extend(
            [
                "--model_filename",
                str(
                    model_filename
                ),
            ]
        )

    if use_autocast:
        command.append(
            "--use_autocast"
        )

    return command


def _run_separator(
    *,
    command: list[str],
    lines: list[str],
    log_callback: LogCallback | None,
) -> int:
    _emit(
        lines,
        log_callback,
        (
            "[Neural Lead/Backing] 실행: "
            + " ".join(
                command
            )
        ),
    )

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            ),
            env=_ffmpeg_env(),
        )
    except OSError as exc:
        raise NeuralLeadBackingError(
            "audio-separator 실행 실패: "
            f"{exc}"
        ) from exc

    assert process.stdout is not None

    for raw_line in process.stdout:
        for part in raw_line.replace(
            "\r",
            "\n",
        ).splitlines():
            if part.strip():
                _emit(
                    lines,
                    log_callback,
                    part,
                )

    return int(
        process.wait()
    )


def _copy_debug(
    *,
    source: Path,
    lead: Path,
    backing: Path,
) -> None:
    cache_root().mkdir(
        parents=True,
        exist_ok=True,
    )

    for src, dst in (
        (
            source,
            debug_source_path(),
        ),
        (
            lead,
            debug_lead_path(),
        ),
        (
            backing,
            debug_backing_path(),
        ),
    ):
        try:
            shutil.copy2(
                src,
                dst,
            )
        except OSError:
            pass


def _write_report(
    lines: list[str],
    data: dict,
) -> None:
    log_dir().mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        log_path().write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    try:
        json_path().write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def separate_lead_backing_for_rvc(
    source_vocal: str | Path,
    *,
    preset: str = DEFAULT_KARAOKE_PRESET,
    use_cache: bool = True,
    use_autocast: bool = True,
    log_callback: LogCallback | None = None,
) -> NeuralLeadBackingResult:
    source = Path(
        source_vocal
    ).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    exe = find_separator_executable()

    if not exe:
        raise NeuralLeadBackingError(
            "audio-separator를 찾지 못했습니다. "
            "SETUP_VOCAL_SEPARATOR_GPU.bat을 먼저 실행하세요."
        )

    preset = str(
        preset
        or DEFAULT_KARAOKE_PRESET
    ).strip()

    lines: list[str] = []

    _emit(
        lines,
        log_callback,
        (
            "[Neural Lead/Backing] 2차 보컬 분리 시작: "
            f"{source.name}"
        ),
    )
    _emit(
        lines,
        log_callback,
        (
            "[Neural Lead/Backing] preset="
            f"{preset}"
        ),
    )
    _emit(
        lines,
        log_callback,
        (
            "[Neural Lead/Backing] 1차 BS-RoFormer vocals를 다시 분리해 "
            "Lead만 RVC에 보내고 Backing/Harmony는 RVC에서 제외합니다."
        ),
    )

    key = _cache_key(
        source,
        preset,
    )
    final_dir = (
        cache_root()
        / key
    )
    final_lead = (
        final_dir
        / "lead.wav"
    )
    final_backing = (
        final_dir
        / "backing.wav"
    )
    metadata_path = (
        final_dir
        / "metadata.json"
    )

    if (
        use_cache
        and final_lead.is_file()
        and final_backing.is_file()
        and final_lead.stat().st_size
        > 1024
        and final_backing.stat().st_size
        > 1024
    ):
        lead_duration, lead_db = (
            _audio_stats(
                final_lead
            )
        )
        backing_duration, backing_db = (
            _audio_stats(
                final_backing
            )
        )
        ratio = _energy_ratio(
            lead_db,
            backing_db,
        )

        _emit(
            lines,
            log_callback,
            (
                "[Neural Lead/Backing] 기존 캐시 사용 / "
                f"Lead RMS={lead_db:.1f} dB / "
                f"Backing RMS={backing_db:.1f} dB"
            ),
        )

        _copy_debug(
            source=source,
            lead=final_lead,
            backing=final_backing,
        )

        report = {
            "status": "cache_hit",
            "source": str(
                source
            ),
            "preset": preset,
            "cache_key": key,
            "lead_path": str(
                final_lead
            ),
            "backing_path": str(
                final_backing
            ),
            "lead_rms_db": lead_db,
            "backing_rms_db": backing_db,
            "lead_energy_ratio": ratio,
            "duration_seconds": lead_duration,
        }
        _write_report(
            lines,
            report,
        )

        return NeuralLeadBackingResult(
            source_vocal=source,
            lead_path=final_lead,
            backing_path=final_backing,
            preset=preset,
            cache_hit=True,
            used_fallback_model=False,
            lead_rms_db=lead_db,
            backing_rms_db=backing_db,
            duration_seconds=lead_duration,
            lead_energy_ratio=ratio,
        )

    model_cache_dir().mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_root().mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="rvc_neural_lead_backing_"
    ) as temp_name:
        temp_root = Path(
            temp_name
        )
        separator_output = (
            temp_root
            / "separator_output"
        )
        separator_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        prepared_temp = None

        try:
            prepared_source, prepared_temp = (
                _prepare_separator_input(
                    source,
                    log_callback=lambda text: (
                        _emit(
                            lines,
                            log_callback,
                            text,
                        )
                    ),
                )
            )

            command = _separator_command(
                exe=exe,
                source=prepared_source,
                output_dir=separator_output,
                preset=preset,
                model_filename=None,
                use_autocast=use_autocast,
            )

            _emit(
                lines,
                log_callback,
                (
                    "[Neural Lead/Backing] Karaoke 3-model ensemble 사용. "
                    "첫 실행이면 모델 다운로드 때문에 오래 걸릴 수 있습니다."
                ),
            )

            return_code = _run_separator(
                command=command,
                lines=lines,
                log_callback=log_callback,
            )

            lead, backing = (
                classify_output_files(
                    separator_output
                )
            )
            used_fallback = False

            if (
                return_code != 0
                or lead is None
                or backing is None
            ):
                _emit(
                    lines,
                    log_callback,
                    (
                        "[Neural Lead/Backing] ensemble 결과가 불완전하여 "
                        "단일 Karaoke RoFormer fallback을 시도합니다."
                    ),
                )

                shutil.rmtree(
                    separator_output,
                    ignore_errors=True,
                )
                separator_output.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                fallback_command = (
                    _separator_command(
                        exe=exe,
                        source=prepared_source,
                        output_dir=separator_output,
                        preset=None,
                        model_filename=DEFAULT_KARAOKE_FALLBACK_MODEL,
                        use_autocast=use_autocast,
                    )
                )

                fallback_code = (
                    _run_separator(
                        command=fallback_command,
                        lines=lines,
                        log_callback=log_callback,
                    )
                )

                lead, backing = (
                    classify_output_files(
                        separator_output
                    )
                )
                used_fallback = True

                if (
                    fallback_code != 0
                    or lead is None
                    or backing is None
                ):
                    listing = "\n".join(
                        str(
                            path.relative_to(
                                separator_output
                            )
                        )
                        for path in separator_output.rglob(
                            "*"
                        )
                        if path.is_file()
                    )

                    raise NeuralLeadBackingError(
                        "Karaoke Lead/Backing 분리 결과를 찾지 못했습니다.\n\n"
                        f"출력:\n{listing or '(없음)'}"
                    )

            assert lead is not None
            assert backing is not None

            lead_duration, lead_db = (
                _audio_stats(
                    lead
                )
            )
            backing_duration, backing_db = (
                _audio_stats(
                    backing
                )
            )

            duration_gap = abs(
                lead_duration
                - backing_duration
            )
            ratio = _energy_ratio(
                lead_db,
                backing_db,
            )

            # Conservative safety gate. A karaoke model can fail on unusual
            # content; do not silently feed an almost-empty stem to RVC.
            if (
                lead_duration
                < 0.5
                or duration_gap
                > max(
                    0.75,
                    lead_duration
                    * 0.02,
                )
                or lead_db
                < -65.0
                or ratio
                < 0.01
            ):
                raise NeuralLeadBackingError(
                    "Karaoke 분리 Lead 품질 게이트를 통과하지 못했습니다. "
                    f"(lead={lead_duration:.2f}s/{lead_db:.1f}dB, "
                    f"backing={backing_duration:.2f}s/{backing_db:.1f}dB, "
                    f"lead_energy={ratio * 100.0:.2f}%)"
                )

            final_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                lead,
                final_lead,
            )
            shutil.copy2(
                backing,
                final_backing,
            )

            metadata = {
                "status": "success",
                "created_unix": time.time(),
                "source": str(
                    source
                ),
                "source_size": int(
                    source.stat().st_size
                ),
                "source_mtime_ns": int(
                    source.stat().st_mtime_ns
                ),
                "preset": preset,
                "fallback_model": (
                    DEFAULT_KARAOKE_FALLBACK_MODEL
                    if used_fallback
                    else None
                ),
                "cache_key": key,
                "lead_path": str(
                    final_lead
                ),
                "backing_path": str(
                    final_backing
                ),
                "lead_duration_seconds": lead_duration,
                "backing_duration_seconds": backing_duration,
                "lead_rms_db": lead_db,
                "backing_rms_db": backing_db,
                "lead_energy_ratio": ratio,
                "pipeline": (
                    "BS-RoFormer vocals -> Karaoke Lead/Backing -> "
                    "Lead RVC / Backing pitch-only"
                ),
            }

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            _copy_debug(
                source=source,
                lead=final_lead,
                backing=final_backing,
            )

            _emit(
                lines,
                log_callback,
                (
                    "[Neural Lead/Backing] 완료 / "
                    f"Lead RMS={lead_db:.1f} dB / "
                    f"Backing RMS={backing_db:.1f} dB / "
                    f"Lead energy={ratio * 100.0:.1f}%"
                ),
            )
            _emit(
                lines,
                log_callback,
                (
                    "[Neural Lead/Backing] Lead debug: "
                    f"{debug_lead_path()}"
                ),
            )
            _emit(
                lines,
                log_callback,
                (
                    "[Neural Lead/Backing] Backing debug: "
                    f"{debug_backing_path()}"
                ),
            )

            _write_report(
                lines,
                metadata,
            )

            return NeuralLeadBackingResult(
                source_vocal=source,
                lead_path=final_lead,
                backing_path=final_backing,
                preset=preset,
                cache_hit=False,
                used_fallback_model=used_fallback,
                lead_rms_db=lead_db,
                backing_rms_db=backing_db,
                duration_seconds=lead_duration,
                lead_energy_ratio=ratio,
            )

        except Exception as exc:
            report = {
                "status": "failed",
                "source": str(
                    source
                ),
                "preset": preset,
                "error_type": type(
                    exc
                ).__name__,
                "error": str(
                    exc
                ),
            }

            _emit(
                lines,
                log_callback,
                (
                    "[Neural Lead/Backing] 실패: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            _write_report(
                lines,
                report,
            )

            if isinstance(
                exc,
                NeuralLeadBackingError,
            ):
                raise

            raise NeuralLeadBackingError(
                str(
                    exc
                )
            ) from exc

        finally:
            if prepared_temp is not None:
                prepared_temp.cleanup()
