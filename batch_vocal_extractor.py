from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from collections import Counter
import hashlib
import shutil

from vocal_separator import (
    DEFAULT_MODEL,
    separate_vocals,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
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


class BatchVocalExtractionError(RuntimeError):
    pass


class BatchVocalExtractionCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class BatchVocalExtractionResult:
    output_dir: Path
    total: int
    completed: int
    skipped: int
    failed: int
    outputs: list[Path]
    failures: list[str]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def batch_log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "batch_vocal_extract_last.log"
    )


def _write_log(
    lines: list[str],
) -> None:
    path = batch_log_path()
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _emit(
    lines: list[str],
    callback: LogCallback | None,
    message: str,
) -> None:
    clean = str(message).strip()

    if not clean:
        return

    lines.append(clean)

    if callback:
        callback(clean)


def _short_source_hash(
    path: Path,
) -> str:
    return hashlib.sha1(
        str(path.resolve()).encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()[:8]


def _build_output_names(
    files: list[Path],
) -> dict[Path, str]:
    stem_counts = Counter(
        path.stem.lower()
        for path in files
    )

    names: dict[Path, str] = {}

    for path in files:
        if stem_counts[path.stem.lower()] > 1:
            suffix = (
                "_"
                + _short_source_hash(path)
            )
        else:
            suffix = ""

        names[path] = (
            f"{path.stem}{suffix}_vocals.wav"
        )

    return names


def normalize_input_files(
    paths: list[str | Path],
) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []

    for value in paths:
        path = Path(
            value
        ).expanduser().resolve()

        if (
            not path.is_file()
            or path.suffix.lower()
            not in SUPPORTED_MEDIA_EXTENSIONS
        ):
            continue

        key = str(path).lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(path)

    return result


def extract_vocals_batch(
    input_files: list[str | Path],
    output_dir: str | Path,
    *,
    model_filename: str = DEFAULT_MODEL,
    use_cache: bool = True,
    use_autocast: bool = True,
    overwrite: bool = False,
    log_callback: LogCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> BatchVocalExtractionResult:
    files = normalize_input_files(
        input_files
    )

    if not files:
        raise BatchVocalExtractionError(
            "보컬 분리할 지원 음원/영상 파일이 없습니다."
        )

    output_root = Path(
        output_dir
    ).expanduser().resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = _build_output_names(
        files
    )

    lines: list[str] = []
    outputs: list[Path] = []
    failures: list[str] = []

    completed = 0
    skipped = 0
    failed = 0

    _emit(
        lines,
        log_callback,
        "여러 곡 보컬 일괄 추출 시작",
    )
    _emit(
        lines,
        log_callback,
        f"입력: {len(files)}개",
    )
    _emit(
        lines,
        log_callback,
        f"출력 폴더: {output_root}",
    )
    _emit(
        lines,
        log_callback,
        f"모델: {model_filename}",
    )
    _emit(
        lines,
        log_callback,
        (
            "처리 방식: 순차 처리 "
            "(GPU VRAM 충돌 방지)"
        ),
    )

    total = len(files)

    for index, source in enumerate(
        files,
        start=1,
    ):
        if (
            cancel_check
            and cancel_check()
        ):
            _emit(
                lines,
                log_callback,
                (
                    "중지 요청 감지: "
                    "현재까지 완료된 결과를 유지하고 종료합니다."
                ),
            )
            _write_log(lines)
            raise BatchVocalExtractionCancelled(
                "사용자가 보컬 일괄 추출을 중지했습니다."
            )

        output = (
            output_root
            / names[source]
        )

        if (
            output.is_file()
            and output.stat().st_size > 0
            and not overwrite
        ):
            skipped += 1
            outputs.append(output)

            message = (
                f"[{index}/{total}] 기존 출력 사용: "
                f"{output.name}"
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
            f"[{index}/{total}] 보컬 분리 중: "
            f"{source.name}"
        )

        _emit(
            lines,
            log_callback,
            ""
        )
        _emit(
            lines,
            log_callback,
            message,
        )

        if progress_callback:
            progress_callback(
                int(
                    (index - 1)
                    / total
                    * 100
                ),
                message,
            )

        resource = None

        try:
            resource = separate_vocals(
                source,
                model_filename=model_filename,
                use_cache=use_cache,
                use_autocast=use_autocast,
                log_callback=lambda text: _emit(
                    lines,
                    log_callback,
                    f"  {text}",
                ),
            )

            shutil.copy2(
                resource.vocal_path,
                output,
            )

            completed += 1
            outputs.append(output)

            _emit(
                lines,
                log_callback,
                (
                    f"[OK] {source.name} "
                    f"-> {output.name}"
                ),
            )

        except Exception as exc:
            failed += 1

            failure = (
                f"{source.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            failures.append(
                failure
            )

            _emit(
                lines,
                log_callback,
                "[FAIL] " + failure,
            )

        finally:
            if resource is not None:
                try:
                    resource.cleanup()
                except Exception:
                    pass

        if progress_callback:
            progress_callback(
                int(
                    index
                    / total
                    * 100
                ),
                (
                    f"{index}/{total} 처리 완료 "
                    f"(성공 {completed}, "
                    f"기존 {skipped}, "
                    f"실패 {failed})"
                ),
            )

        _write_log(lines)

    _emit(
        lines,
        log_callback,
        ""
    )
    _emit(
        lines,
        log_callback,
        (
            "일괄 추출 완료: "
            f"새로 생성 {completed}, "
            f"기존 사용 {skipped}, "
            f"실패 {failed}"
        ),
    )
    _write_log(lines)

    return BatchVocalExtractionResult(
        output_dir=output_root,
        total=total,
        completed=completed,
        skipped=skipped,
        failed=failed,
        outputs=outputs,
        failures=failures,
    )
