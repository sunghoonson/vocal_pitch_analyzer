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

from audio_transposer import (
    AudioTransposeError,
    find_ffmpeg,
    transpose_audio,
)
from vocal_separator import (
    _ffmpeg_env,
    _prepare_separator_input,
    find_separator_executable,
    model_cache_dir,
)


# V30_INSTRUMENT_SMART_SHIFT_PATCH

DEFAULT_INSTRUMENT_MODEL = "htdemucs_ft.yaml"

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]


class InstrumentSmartShiftError(RuntimeError):
    pass


@dataclass(slots=True)
class InstrumentStemSet:
    drums: Path
    bass: Path
    other: Path
    residual: Path
    model_filename: str
    cache_hit: bool


@dataclass(slots=True)
class InstrumentSmartShiftResult:
    output_path: Path
    smart_used: bool
    fallback_used: bool
    preserve_drums: bool
    model_filename: str | None
    stem_cache_hit: bool


def project_root() -> Path:
    return Path(__file__).resolve().parent


def cache_root() -> Path:
    return (
        project_root()
        / "cache"
        / "instrument_stems"
    )


def log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "instrument_smart_shift_last.log"
    )


def report_path() -> Path:
    return (
        project_root()
        / "logs"
        / "instrument_smart_shift_last.json"
    )


def _emit(
    callback: LogCallback | None,
    lines: list[str],
    message: str,
) -> None:
    clean = str(message).strip()

    if not clean:
        return

    lines.append(clean)

    if callback is not None:
        callback(clean)


def _progress(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
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


def _write_logs(
    lines: list[str],
    payload: dict,
) -> None:
    try:
        log_path().parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path().write_text(
            "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    try:
        report_path().parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report_path().write_text(
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


def _cache_key(
    input_path: Path,
    model_filename: str,
) -> str:
    stat = input_path.stat()

    payload = {
        "path": str(
            input_path.resolve()
        ),
        "size": int(
            stat.st_size
        ),
        "mtime_ns": int(
            stat.st_mtime_ns
        ),
        "model": str(
            model_filename
        ),
        "schema": 1,
    }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[
        :24
    ]


def _find_named_stem(
    folder: Path,
    token: str,
) -> Path | None:
    token = str(token).lower()

    for candidate in (
        folder
        / f"{token}.wav",
        folder
        / f"{token.capitalize()}.wav",
    ):
        if (
            candidate.is_file()
            and candidate.stat().st_size > 0
        ):
            return candidate

    for candidate in sorted(
        folder.glob(
            "*.wav"
        )
    ):
        if (
            token
            in candidate.stem.lower()
            and candidate.stat().st_size > 0
        ):
            return candidate

    return None


def _valid_cached_stems(
    folder: Path,
) -> InstrumentStemSet | None:
    drums = _find_named_stem(
        folder,
        "drums",
    )
    bass = _find_named_stem(
        folder,
        "bass",
    )
    other = _find_named_stem(
        folder,
        "other",
    )
    residual = _find_named_stem(
        folder,
        "residual",
    )

    if all(
        path is not None
        for path in (
            drums,
            bass,
            other,
            residual,
        )
    ):
        return InstrumentStemSet(
            drums=drums,
            bass=bass,
            other=other,
            residual=residual,
            model_filename=DEFAULT_INSTRUMENT_MODEL,
            cache_hit=True,
        )

    return None


def separate_instrumental_stems(
    instrumental_path: str | Path,
    *,
    model_filename: str = DEFAULT_INSTRUMENT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    log_callback: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> InstrumentStemSet:
    """
    Run Demucs on the already BS-RoFormer-separated instrumental stem.

    This is intentional:
      - BS-RoFormer remains the high quality vocal/instrumental boundary.
      - Demucs is used only to classify that instrumental into
        drums / bass / other / residual.
      - The Demucs 'Vocals' output is named residual because the input
        is already an instrumental stem; it may contain bleed or
        material Demucs classified as vocal-like. It is not discarded.
    """
    source = Path(
        instrumental_path
    ).resolve()

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    exe = find_separator_executable()

    if not exe:
        raise InstrumentSmartShiftError(
            "audio-separator를 찾을 수 없습니다. "
            "SETUP_VOCAL_SEPARATOR_GPU.bat을 먼저 실행하세요."
        )

    lines: list[str] = []

    key = _cache_key(
        source,
        model_filename,
    )
    final_dir = (
        cache_root()
        / key
    )

    if use_cache:
        cached = _valid_cached_stems(
            final_dir
        )

        if cached is not None:
            cached.model_filename = (
                model_filename
            )
            _emit(
                log_callback,
                lines,
                "[Instrument Smart Shift] Demucs 4-stem 캐시 사용",
            )
            _write_logs(
                lines,
                {
                    "status": "cache_hit",
                    "source": str(source),
                    "model": model_filename,
                    "cache_dir": str(final_dir),
                },
            )
            _progress(
                progress,
                45,
                "Demucs 악기 stem 캐시 사용",
            )
            return cached

    model_dir = model_cache_dir()
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cache_root().mkdir(
        parents=True,
        exist_ok=True,
    )

    work_parent = (
        cache_root()
        / "_working"
    )
    work_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _progress(
        progress,
        0,
        "Demucs 4-stem 악기 분리 준비 중...",
    )

    with tempfile.TemporaryDirectory(
        prefix=f"{key}_",
        dir=str(work_parent),
    ) as temp_name:
        work_dir = Path(
            temp_name
        )

        separator_temp = None

        try:
            separator_input, separator_temp = (
                _prepare_separator_input(
                    source,
                    log_callback=lambda text: (
                        _emit(
                            log_callback,
                            lines,
                            text,
                        )
                    ),
                )
            )

            custom_names = json.dumps(
                {
                    "Vocals": "residual",
                    "Drums": "drums",
                    "Bass": "bass",
                    "Other": "other",
                },
                ensure_ascii=False,
            )

            command = [
                exe,
                str(
                    separator_input
                ),
                "--model_filename",
                model_filename,
                "--output_format",
                "WAV",
                "--output_dir",
                str(
                    work_dir
                ),
                "--model_file_dir",
                str(
                    model_dir
                ),
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
                lines,
                (
                    "[Instrument Smart Shift] "
                    f"Demucs 모델: {model_filename}"
                ),
            )
            _emit(
                log_callback,
                lines,
                (
                    "[Instrument Smart Shift] "
                    "BS-RoFormer instrumental을 "
                    "Drums/Bass/Other/Residual로 분리"
                ),
            )
            _emit(
                log_callback,
                lines,
                (
                    "[Instrument Smart Shift] "
                    "첫 실행이면 Demucs 모델 다운로드로 시간이 더 걸릴 수 있습니다."
                ),
            )

            creationflags = getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )

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

            assert process.stdout is not None

            progress_value = 4

            for raw_line in process.stdout:
                for part in raw_line.replace(
                    "\r",
                    "\n",
                ).splitlines():
                    clean = part.strip()

                    if not clean:
                        continue

                    _emit(
                        log_callback,
                        lines,
                        clean,
                    )

                    # audio-separator does not expose a stable numerical
                    # percentage for all Demucs versions. Give the GUI a
                    # monotonic activity indication while preserving room
                    # for the later pitch-shift stages.
                    progress_value = min(
                        42,
                        progress_value
                        + 1,
                    )
                    _progress(
                        progress,
                        progress_value,
                        "Demucs 악기 분리 중...",
                    )

            return_code = process.wait()

            if return_code != 0:
                raise InstrumentSmartShiftError(
                    "Demucs 악기 분리에 실패했습니다.\n\n"
                    + "\n".join(
                        lines[
                            -80:
                        ]
                    )[
                        -7000:
                    ]
                )

            drums = _find_named_stem(
                work_dir,
                "drums",
            )
            bass = _find_named_stem(
                work_dir,
                "bass",
            )
            other = _find_named_stem(
                work_dir,
                "other",
            )
            residual = _find_named_stem(
                work_dir,
                "residual",
            )

            missing = [
                name
                for name, path
                in (
                    (
                        "drums",
                        drums,
                    ),
                    (
                        "bass",
                        bass,
                    ),
                    (
                        "other",
                        other,
                    ),
                    (
                        "residual",
                        residual,
                    ),
                )
                if path is None
            ]

            if missing:
                listing = "\n".join(
                    path.name
                    for path in work_dir.iterdir()
                    if path.is_file()
                )

                raise InstrumentSmartShiftError(
                    "Demucs 분리는 종료됐지만 필요한 stem을 찾지 못했습니다.\n"
                    f"누락: {', '.join(missing)}\n\n"
                    f"출력:\n{listing or '(없음)'}"
                )

            assert drums is not None
            assert bass is not None
            assert other is not None
            assert residual is not None

            final_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            final_paths = {
                "drums": (
                    final_dir
                    / "drums.wav"
                ),
                "bass": (
                    final_dir
                    / "bass.wav"
                ),
                "other": (
                    final_dir
                    / "other.wav"
                ),
                "residual": (
                    final_dir
                    / "residual.wav"
                ),
            }

            for key_name, source_path in (
                (
                    "drums",
                    drums,
                ),
                (
                    "bass",
                    bass,
                ),
                (
                    "other",
                    other,
                ),
                (
                    "residual",
                    residual,
                ),
            ):
                shutil.copy2(
                    source_path,
                    final_paths[
                        key_name
                    ],
                )

            metadata = {
                "source": str(
                    source
                ),
                "source_size": int(
                    source.stat().st_size
                ),
                "source_mtime_ns": int(
                    source.stat().st_mtime_ns
                ),
                "model": model_filename,
                "created_unix": time.time(),
                "pipeline": (
                    "BS-RoFormer instrumental -> "
                    "Demucs drums/bass/other/residual"
                ),
            }

            (
                final_dir
                / "metadata.json"
            ).write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            _progress(
                progress,
                45,
                "Demucs 4-stem 악기 분리 완료",
            )

            _write_logs(
                lines,
                {
                    "status": "success",
                    **metadata,
                    "cache_dir": str(
                        final_dir
                    ),
                },
            )

            return InstrumentStemSet(
                drums=final_paths[
                    "drums"
                ],
                bass=final_paths[
                    "bass"
                ],
                other=final_paths[
                    "other"
                ],
                residual=final_paths[
                    "residual"
                ],
                model_filename=model_filename,
                cache_hit=False,
            )

        finally:
            if separator_temp is not None:
                separator_temp.cleanup()


def _mix_wav_inputs(
    inputs: list[Path],
    output_path: Path,
) -> Path:
    if not inputs:
        raise InstrumentSmartShiftError(
            "재합성할 악기 stem이 없습니다."
        )

    ffmpeg = find_ffmpeg()

    if not ffmpeg:
        raise InstrumentSmartShiftError(
            "악기 stem 재합성에 필요한 FFmpeg를 찾지 못했습니다."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]

    for path in inputs:
        command.extend(
            [
                "-i",
                str(
                    path
                ),
            ]
        )

    input_labels = "".join(
        f"[{index}:a]"
        for index in range(
            len(
                inputs
            )
        )
    )

    command.extend(
        [
            "-filter_complex",
            (
                input_labels
                + f"amix=inputs={len(inputs)}:"
                "duration=longest:normalize=0[mix]"
            ),
            "-map",
            "[mix]",
            "-vn",
            "-c:a",
            "pcm_f32le",
            str(
                output_path
            ),
        ]
    )

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
        or not output_path.is_file()
        or output_path.stat().st_size <= 0
    ):
        raise InstrumentSmartShiftError(
            "악기 stem 재합성에 실패했습니다.\n\n"
            + (
                completed.stderr
                or completed.stdout
                or ""
            )[
                -5000:
            ]
        )

    return output_path


def _shift_one(
    source: Path,
    output: Path,
    *,
    semitones: int,
    progress: ProgressCallback | None,
    start_percent: int,
    end_percent: int,
    label: str,
) -> Path:
    span = max(
        1,
        end_percent
        - start_percent,
    )

    def local_progress(
        percent: int,
        text: str,
    ) -> None:
        mapped = (
            start_percent
            + int(
                max(
                    0,
                    min(
                        100,
                        percent,
                    ),
                )
                * span
                / 100.0
            )
        )

        _progress(
            progress,
            mapped,
            f"{label}: {text}",
        )

    try:
        return transpose_audio(
            source,
            output,
            semitones=semitones,
            preserve_formant=False,
            quality="quality",
            progress=local_progress,
        )
    except AudioTransposeError as exc:
        raise InstrumentSmartShiftError(
            f"{label} 키 변경 실패:\n{exc}"
        ) from exc


def build_smart_shifted_instrumental(
    instrumental_path: str | Path,
    output_path: str | Path,
    *,
    semitones: int,
    preserve_drums: bool = True,
    model_filename: str = DEFAULT_INSTRUMENT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    progress: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> InstrumentSmartShiftResult:
    source = Path(
        instrumental_path
    ).resolve()
    output = Path(
        output_path
    ).resolve()

    lines: list[str] = []

    stems = separate_instrumental_stems(
        source,
        model_filename=model_filename,
        use_cache=use_cache,
        use_autocast=use_autocast,
        log_callback=lambda text: (
            _emit(
                log_callback,
                lines,
                text,
            )
        ),
        progress=progress,
    )

    with tempfile.TemporaryDirectory(
        prefix="instrument_smart_shift_"
    ) as temp_name:
        temp = Path(
            temp_name
        )

        shifted_bass = _shift_one(
            stems.bass,
            temp
            / "bass_shifted.wav",
            semitones=semitones,
            progress=progress,
            start_percent=46,
            end_percent=59,
            label="Bass",
        )

        shifted_other = _shift_one(
            stems.other,
            temp
            / "other_shifted.wav",
            semitones=semitones,
            progress=progress,
            start_percent=59,
            end_percent=73,
            label="Other",
        )

        shifted_residual = _shift_one(
            stems.residual,
            temp
            / "residual_shifted.wav",
            semitones=semitones,
            progress=progress,
            start_percent=73,
            end_percent=86,
            label="Residual",
        )

        if preserve_drums:
            drums_for_mix = (
                stems.drums
            )
            _emit(
                log_callback,
                lines,
                (
                    "[Instrument Smart Shift] "
                    "Drums/Percussion은 원래 pitch 유지"
                ),
            )
        else:
            drums_for_mix = _shift_one(
                stems.drums,
                temp
                / "drums_shifted.wav",
                semitones=semitones,
                progress=progress,
                start_percent=86,
                end_percent=93,
                label="Drums",
            )
            _emit(
                log_callback,
                lines,
                (
                    "[Instrument Smart Shift] "
                    "Drums도 동일 semitone으로 이동"
                ),
            )

        _progress(
            progress,
            94,
            "악기 stem 재합성 중...",
        )

        _mix_wav_inputs(
            [
                drums_for_mix,
                shifted_bass,
                shifted_other,
                shifted_residual,
            ],
            output,
        )

    _progress(
        progress,
        100,
        "Instrument Smart Shift 완료",
    )

    payload = {
        "status": "success",
        "source": str(
            source
        ),
        "output": str(
            output
        ),
        "semitones": int(
            semitones
        ),
        "preserve_drums": bool(
            preserve_drums
        ),
        "model": model_filename,
        "stem_cache_hit": bool(
            stems.cache_hit
        ),
        "strategy": {
            "drums": (
                "original pitch"
                if preserve_drums
                else f"{semitones:+d} semitone"
            ),
            "bass": f"{semitones:+d} semitone",
            "other": f"{semitones:+d} semitone",
            "residual": f"{semitones:+d} semitone",
        },
    }

    _write_logs(
        lines,
        payload,
    )

    return InstrumentSmartShiftResult(
        output_path=output,
        smart_used=True,
        fallback_used=False,
        preserve_drums=bool(
            preserve_drums
        ),
        model_filename=model_filename,
        stem_cache_hit=bool(
            stems.cache_hit
        ),
    )


def prepare_shifted_instrumental(
    instrumental_path: str | Path,
    output_path: str | Path,
    *,
    semitones: int,
    smart_enabled: bool = True,
    preserve_drums: bool = True,
    model_filename: str = DEFAULT_INSTRUMENT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    fallback_legacy: bool = True,
    progress: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> InstrumentSmartShiftResult:
    source = Path(
        instrumental_path
    ).resolve()
    output = Path(
        output_path
    ).resolve()

    if not smart_enabled:
        try:
            transpose_audio(
                source,
                output,
                semitones=semitones,
                preserve_formant=False,
                quality="quality",
                progress=progress,
            )
        except AudioTransposeError as exc:
            raise InstrumentSmartShiftError(
                str(
                    exc
                )
            ) from exc

        return InstrumentSmartShiftResult(
            output_path=output,
            smart_used=False,
            fallback_used=False,
            preserve_drums=False,
            model_filename=None,
            stem_cache_hit=False,
        )

    try:
        return build_smart_shifted_instrumental(
            source,
            output,
            semitones=semitones,
            preserve_drums=preserve_drums,
            model_filename=model_filename,
            use_cache=use_cache,
            use_autocast=use_autocast,
            progress=progress,
            log_callback=log_callback,
        )

    except Exception as exc:
        if not fallback_legacy:
            raise

        message = (
            "[Instrument Smart Shift] 실패 - "
            "전체 변환을 중단하지 않고 기존 반주 전체 RubberBand 방식으로 fallback: "
            f"{type(exc).__name__}: {exc}"
        )

        if log_callback is not None:
            log_callback(
                message
            )

        try:
            lines = [
                message,
            ]
            _write_logs(
                lines,
                {
                    "status": "fallback",
                    "source": str(
                        source
                    ),
                    "output": str(
                        output
                    ),
                    "semitones": int(
                        semitones
                    ),
                    "error_type": type(
                        exc
                    ).__name__,
                    "error": str(
                        exc
                    ),
                    "fallback": "legacy whole-instrumental RubberBand",
                },
            )
        except Exception:
            pass

        try:
            transpose_audio(
                source,
                output,
                semitones=semitones,
                preserve_formant=False,
                quality="quality",
                progress=progress,
            )
        except AudioTransposeError as fallback_exc:
            raise InstrumentSmartShiftError(
                "Smart Shift와 기존 반주 Pitch Shift가 모두 실패했습니다.\n\n"
                f"Smart: {exc}\n\n"
                f"Legacy: {fallback_exc}"
            ) from fallback_exc

        return InstrumentSmartShiftResult(
            output_path=output,
            smart_used=False,
            fallback_used=True,
            preserve_drums=False,
            model_filename=model_filename,
            stem_cache_hit=False,
        )
