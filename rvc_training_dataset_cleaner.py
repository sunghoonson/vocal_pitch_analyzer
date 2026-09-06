from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from collections import Counter
import hashlib
import json
import shutil
import time

from rvc_lead_selector import (
    LeadSelectorReport,
    select_lead_vocal,
)


# V28_TRAINING_LEAD_DATASET_CLEANER_PATCH

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]

MANIFEST_NAME = ".rvc_lead_clean_manifest.json"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}


class RVCTrainingDatasetCleanError(RuntimeError):
    pass


class RVCTrainingDatasetCleanCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class TrainingDatasetCleanResult:
    source_dir: Path
    output_dir: Path
    review_dir: Path
    strength: str
    total: int
    accepted: int
    cached: int
    review: int
    failed: int
    outputs: list[Path]
    review_files: list[Path]
    manifest_path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def cleaner_log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_training_lead_cleaner_last.log"
    )


def cleaner_json_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_training_lead_cleaner_last.json"
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


def _write_log(
    lines: list[str],
) -> None:
    path = cleaner_log_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "\n".join(
                lines
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _write_summary(
    payload: dict,
) -> None:
    path = cleaner_json_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
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


def dataset_audio_files(
    folder: str | Path,
) -> list[Path]:
    root = Path(
        folder
    ).expanduser().resolve()

    if not root.is_dir():
        return []

    return sorted(
        [
            path
            for path in root.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_AUDIO_EXTENSIONS
            )
        ],
        key=lambda path: (
            path.name.lower()
        ),
    )


def _read_manifest(
    folder: str | Path,
) -> dict | None:
    path = (
        Path(
            folder
        ).expanduser().resolve()
        / MANIFEST_NAME
    )

    if not path.is_file():
        return None

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            return None

        return data

    except Exception:
        return None


def is_lead_clean_dataset(
    folder: str | Path,
) -> bool:
    data = _read_manifest(
        folder
    )

    return bool(
        data
        and data.get(
            "cleaner_version"
        )
        == "2.8"
        and data.get(
            "source_dir"
        )
    )


def cleaning_source_dir(
    selected_folder: str | Path,
) -> Path:
    selected = Path(
        selected_folder
    ).expanduser().resolve()

    data = _read_manifest(
        selected
    )

    if data:
        raw_source = str(
            data.get(
                "source_dir",
                "",
            )
        ).strip()

        if raw_source:
            candidate = Path(
                raw_source
            ).expanduser().resolve()

            if candidate.is_dir():
                return candidate

    return selected


def default_clean_output_dir(
    selected_folder: str | Path,
) -> Path:
    selected = Path(
        selected_folder
    ).expanduser().resolve()

    if is_lead_clean_dataset(
        selected
    ):
        return selected

    name = selected.name
    lower = name.lower()

    if lower == "_rvc_vocals":
        output_name = (
            "_rvc_lead_vocals"
        )
    elif lower.endswith(
        "_rvc_vocals"
    ):
        output_name = (
            name[
                :-len(
                    "_rvc_vocals"
                )
            ]
            + "_rvc_lead_vocals"
        )
    elif lower.endswith(
        "_vocals"
    ):
        output_name = (
            name[
                :-len(
                    "_vocals"
                )
            ]
            + "_lead_vocals"
        )
    else:
        output_name = (
            name
            + "_lead"
        )

    return (
        selected.parent
        / output_name
    )


def default_review_dir(
    output_dir: str | Path,
) -> Path:
    output = Path(
        output_dir
    ).expanduser().resolve()

    return (
        output.parent
        / (
            output.name
            + "_review"
        )
    )


def _source_signature(
    path: Path,
) -> dict[str, int]:
    stat = path.stat()

    return {
        "size": int(
            stat.st_size
        ),
        "mtime_ns": int(
            stat.st_mtime_ns
        ),
    }


def _short_hash(
    path: Path,
) -> str:
    return hashlib.sha1(
        str(
            path.resolve()
        ).encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[
        :8
    ]


def _output_names(
    files: list[Path],
) -> dict[Path, str]:
    counts = Counter(
        path.stem.lower()
        for path in files
    )

    result: dict[
        Path,
        str,
    ] = {}

    for source in files:
        suffix = (
            "_"
            + _short_hash(
                source
            )
            if counts[
                source.stem.lower()
            ]
            > 1
            else ""
        )

        result[
            source
        ] = (
            f"{source.stem}{suffix}_lead.wav"
        )

    return result


def _quality_thresholds(
    strength: str,
) -> dict[str, float]:
    key = (
        strength
        if strength
        in {
            "gentle",
            "balanced",
            "strict",
        }
        else "balanced"
    )

    return {
        "gentle": {
            "min_selected_ratio": 0.025,
            "min_energy_ratio": 0.0025,
            "min_confidence": 0.17,
        },
        "balanced": {
            "min_selected_ratio": 0.040,
            "min_energy_ratio": 0.0050,
            "min_confidence": 0.21,
        },
        "strict": {
            "min_selected_ratio": 0.030,
            "min_energy_ratio": 0.0035,
            "min_confidence": 0.22,
        },
    }[
        key
    ]


def _quality_check(
    report: LeadSelectorReport,
    strength: str,
) -> tuple[
    bool,
    str,
]:
    limits = _quality_thresholds(
        strength
    )

    minimum_seconds = max(
        1.5,
        float(
            report.duration_seconds
        )
        * 0.02,
    )

    reasons: list[str] = []

    if (
        report.selected_seconds
        < minimum_seconds
    ):
        reasons.append(
            (
                "Lead 선택 시간이 너무 짧음 "
                f"({report.selected_seconds:.1f}s)"
            )
        )

    if (
        report.selected_ratio
        < limits[
            "min_selected_ratio"
        ]
    ):
        reasons.append(
            (
                "Lead 선택 비율 낮음 "
                f"({report.selected_ratio * 100.0:.1f}%)"
            )
        )

    if (
        report.lead_energy_ratio
        < limits[
            "min_energy_ratio"
        ]
    ):
        reasons.append(
            (
                "Lead 에너지 낮음 "
                f"({report.lead_energy_ratio * 100.0:.2f}%)"
            )
        )

    if (
        report.mean_lead_confidence
        < limits[
            "min_confidence"
        ]
    ):
        reasons.append(
            (
                "Lead confidence 낮음 "
                f"({report.mean_lead_confidence:.3f})"
            )
        )

    if reasons:
        return (
            False,
            "; ".join(
                reasons
            ),
        )

    return (
        True,
        "quality gate 통과",
    )


def _manifest_record_matches(
    record: dict,
    source: Path,
    output: Path,
    *,
    strength: str,
) -> bool:
    if not isinstance(
        record,
        dict,
    ):
        return False

    if (
        record.get(
            "status"
        )
        not in {
            "accepted",
            "cached",
        }
    ):
        return False

    if (
        str(
            record.get(
                "strength",
                "",
            )
        )
        != strength
    ):
        return False

    signature = _source_signature(
        source
    )

    if (
        int(
            record.get(
                "source_size",
                -1,
            )
        )
        != signature[
            "size"
        ]
    ):
        return False

    if (
        int(
            record.get(
                "source_mtime_ns",
                -1,
            )
        )
        != signature[
            "mtime_ns"
        ]
    ):
        return False

    return bool(
        output.is_file()
        and output.stat().st_size
        > 0
    )


def clean_training_dataset(
    selected_dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    strength: str = "balanced",
    overwrite: bool = False,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> TrainingDatasetCleanResult:
    selected = Path(
        selected_dataset_dir
    ).expanduser().resolve()

    raw_source = cleaning_source_dir(
        selected
    )

    if not raw_source.is_dir():
        raise RVCTrainingDatasetCleanError(
            "학습용 원본 데이터셋 폴더를 찾을 수 없습니다."
        )

    strength = (
        strength
        if strength
        in {
            "gentle",
            "balanced",
            "strict",
        }
        else "balanced"
    )

    output_root = (
        Path(
            output_dir
        ).expanduser().resolve()
        if output_dir
        is not None
        else default_clean_output_dir(
            selected
        )
    )

    if (
        output_root
        == raw_source
    ):
        raise RVCTrainingDatasetCleanError(
            "Lead Cleaner 출력 폴더가 원본 데이터셋과 같습니다. "
            "원본 보호를 위해 정제를 중단합니다."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_root = default_review_dir(
        output_root
    )
    review_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = dataset_audio_files(
        raw_source
    )

    if not files:
        raise RVCTrainingDatasetCleanError(
            "원본 데이터셋 바로 아래에 지원 음성 파일이 없습니다."
        )

    names = _output_names(
        files
    )

    old_manifest = _read_manifest(
        output_root
    ) or {}

    old_records_by_source: dict[
        str,
        dict,
    ] = {}

    for record in old_manifest.get(
        "files",
        [],
    ):
        if not isinstance(
            record,
            dict,
        ):
            continue

        key = str(
            record.get(
                "source_path",
                "",
            )
        ).lower()

        if key:
            old_records_by_source[
                key
            ] = record

    lines: list[str] = []
    records: list[dict] = []
    outputs: list[Path] = []
    review_files: list[Path] = []

    accepted = 0
    cached = 0
    review_count = 0
    failed = 0

    _emit(
        lines,
        log_callback,
        "RVC 학습용 Lead Vocal Dataset Cleaner 시작",
    )
    _emit(
        lines,
        log_callback,
        f"원본 데이터셋: {raw_source}",
    )
    _emit(
        lines,
        log_callback,
        f"정제 데이터셋: {output_root}",
    )
    _emit(
        lines,
        log_callback,
        f"검토 폴더: {review_root}",
    )
    _emit(
        lines,
        log_callback,
        f"Lead 선별 강도: {strength}",
    )
    _emit(
        lines,
        log_callback,
        (
            "원본 파일은 수정/삭제하지 않습니다. "
            "quality gate 실패 파일은 학습 폴더에 넣지 않습니다."
        ),
    )

    expected_output_names: set[
        str
    ] = set()

    total = len(
        files
    )

    for index, source in enumerate(
        files,
        start=1,
    ):
        if (
            cancel_check
            and cancel_check()
        ):
            _write_log(
                lines
            )
            raise RVCTrainingDatasetCleanCancelled(
                "사용자가 학습용 Lead Dataset 정제를 중지했습니다."
            )

        output = (
            output_root
            / names[
                source
            ]
        )
        expected_output_names.add(
            output.name.lower()
        )

        record_key = str(
            source.resolve()
        ).lower()
        old_record = old_records_by_source.get(
            record_key,
            {},
        )

        if (
            not overwrite
            and _manifest_record_matches(
                old_record,
                source,
                output,
                strength=strength,
            )
        ):
            cached += 1
            outputs.append(
                output
            )

            cached_record = dict(
                old_record
            )
            cached_record[
                "status"
            ] = "cached"
            records.append(
                cached_record
            )

            message = (
                f"[{index}/{total}] [CACHE] "
                f"{source.name} -> {output.name}"
            )
            _emit(
                lines,
                log_callback,
                message,
            )

            if progress_callback:
                progress_callback(
                    int(
                        index
                        / total
                        * 100
                    ),
                    message,
                )

            continue

        message = (
            f"[{index}/{total}] Lead 정제 중: "
            f"{source.name}"
        )
        _emit(
            lines,
            log_callback,
            message,
        )

        if progress_callback:
            progress_callback(
                int(
                    (
                        index
                        - 1
                    )
                    / total
                    * 100
                ),
                message,
            )

        with tempfile_directory(
            prefix=(
                "rvc_training_lead_"
                + _short_hash(
                    source
                )
                + "_"
            )
        ) as temp:
            temp_lead = (
                temp
                / "lead.wav"
            )
            temp_residual = (
                temp
                / "residual.wav"
            )

            signature = _source_signature(
                source
            )

            try:
                report = select_lead_vocal(
                    source,
                    temp_lead,
                    temp_residual,
                    strength=strength,
                    log_callback=lambda text: (
                        _emit(
                            lines,
                            log_callback,
                            "  "
                            + text,
                        )
                    ),
                    save_debug_copy=False,
                )

                passed, reason = (
                    _quality_check(
                        report,
                        strength,
                    )
                )

                base_record = {
                    "source_path": str(
                        source
                    ),
                    "source_name": source.name,
                    "source_size": signature[
                        "size"
                    ],
                    "source_mtime_ns": signature[
                        "mtime_ns"
                    ],
                    "strength": strength,
                    "selector_report": asdict(
                        report
                    ),
                    "quality_reason": reason,
                    "output_path": str(
                        output
                    ),
                }

                if passed:
                    shutil.copy2(
                        temp_lead,
                        output,
                    )

                    accepted += 1
                    outputs.append(
                        output
                    )

                    base_record[
                        "status"
                    ] = "accepted"
                    records.append(
                        base_record
                    )

                    _emit(
                        lines,
                        log_callback,
                        (
                            "  [ACCEPT] "
                            f"Lead {report.selected_ratio * 100.0:.1f}% / "
                            f"energy {report.lead_energy_ratio * 100.0:.1f}% / "
                            f"conf {report.mean_lead_confidence:.3f}"
                        ),
                    )

                else:
                    review_count += 1

                    if output.exists():
                        try:
                            output.unlink()
                        except OSError:
                            pass

                    review_target = (
                        review_root
                        / source.name
                    )

                    shutil.copy2(
                        source,
                        review_target,
                    )
                    review_files.append(
                        review_target
                    )

                    base_record[
                        "status"
                    ] = "review"
                    base_record[
                        "review_path"
                    ] = str(
                        review_target
                    )
                    records.append(
                        base_record
                    )

                    _emit(
                        lines,
                        log_callback,
                        (
                            "  [REVIEW] 학습에서 제외: "
                            + reason
                        ),
                    )

            except Exception as exc:
                failed += 1

                if output.exists():
                    try:
                        output.unlink()
                    except OSError:
                        pass

                review_target = (
                    review_root
                    / source.name
                )

                try:
                    shutil.copy2(
                        source,
                        review_target,
                    )
                    review_files.append(
                        review_target
                    )
                except OSError:
                    pass

                records.append(
                    {
                        "source_path": str(
                            source
                        ),
                        "source_name": source.name,
                        "source_size": signature[
                            "size"
                        ],
                        "source_mtime_ns": signature[
                            "mtime_ns"
                        ],
                        "strength": strength,
                        "status": "failed",
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "output_path": str(
                            output
                        ),
                    }
                )

                _emit(
                    lines,
                    log_callback,
                    (
                        "  [FAIL] "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        if progress_callback:
            progress_callback(
                int(
                    index
                    / total
                    * 100
                ),
                (
                    f"{index}/{total} 정제 완료 "
                    f"(채택 {accepted}, 캐시 {cached}, "
                    f"검토 {review_count}, 실패 {failed})"
                ),
            )

        _write_log(
            lines
        )

    # Remove stale top-level audio outputs that no longer map to
    # a current raw source file. This prevents old stems from
    # silently remaining in the training dataset.
    removed_stale = 0

    for path in dataset_audio_files(
        output_root
    ):
        if (
            path.name.lower()
            not in expected_output_names
        ):
            try:
                path.unlink()
                removed_stale += 1
            except OSError:
                pass

    usable = accepted + cached

    manifest = {
        "cleaner_version": "2.8",
        "created_at_unix": time.time(),
        "source_dir": str(
            raw_source
        ),
        "output_dir": str(
            output_root
        ),
        "review_dir": str(
            review_root
        ),
        "strength": strength,
        "total": total,
        "usable": usable,
        "accepted": accepted,
        "cached": cached,
        "review": review_count,
        "failed": failed,
        "removed_stale": removed_stale,
        "files": records,
    }

    manifest_path = (
        output_root
        / MANIFEST_NAME
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_summary(
        manifest
    )

    _emit(
        lines,
        log_callback,
        "",
    )
    _emit(
        lines,
        log_callback,
        (
            "Lead Dataset 정제 완료: "
            f"학습 사용 {usable}/{total}, "
            f"검토 {review_count}, 실패 {failed}"
        ),
    )
    _emit(
        lines,
        log_callback,
        f"학습 데이터셋: {output_root}",
    )

    if (
        review_count
        or failed
    ):
        _emit(
            lines,
            log_callback,
            (
                "검토 필요 원본: "
                f"{review_root}"
            ),
        )

    _write_log(
        lines
    )

    if usable <= 0:
        raise RVCTrainingDatasetCleanError(
            "Lead Cleaner가 학습에 사용할 수 있는 파일을 하나도 만들지 못했습니다. "
            "강도를 낮추거나 review 폴더를 확인하세요."
        )

    return TrainingDatasetCleanResult(
        source_dir=raw_source,
        output_dir=output_root,
        review_dir=review_root,
        strength=strength,
        total=total,
        accepted=accepted,
        cached=cached,
        review=review_count,
        failed=failed,
        outputs=outputs,
        review_files=review_files,
        manifest_path=manifest_path,
    )


class tempfile_directory:
    def __init__(
        self,
        *,
        prefix: str,
    ):
        import tempfile

        self.path = Path(
            tempfile.mkdtemp(
                prefix=prefix
            )
        )

    def __enter__(
        self,
    ) -> Path:
        return self.path

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        shutil.rmtree(
            self.path,
            ignore_errors=True,
        )
