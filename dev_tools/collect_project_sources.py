from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path


DEFAULT_SETTINGS = {
    "output_mode": "sibling",
    "output_dir_name": "_project_snapshots",
    "include_extensions": [
        ".py", ".bat", ".cmd", ".ps1",
        ".json", ".toml", ".yaml", ".yml",
        ".md", ".txt", ".ini", ".cfg",
        ".spec", ".ui"
    ],
    "include_filenames": [
        ".gitignore",
        ".gitattributes",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml"
    ],
    "exclude_dirs": [
        ".git",
        ".venv",
        ".venv_separator",
        ".venv_svc",
        ".venv_rvc",
        "seed-vc",
        "rvc",
        "rvc_models",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "cache",
        "models",
        "logs",
        "build",
        "dist",
        "project_snapshots",
        "_project_snapshots"
    ],
    "exclude_dir_globs": [
        "backup_before_*",
        "backup_*"
    ],
    "exclude_file_globs": [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.dll",
        "*.exe",
        "*.ckpt",
        "*.onnx",
        "*.pth",
        "*.pt",
        "*.safetensors",
        "*.wav",
        "*.mp3",
        "*.m4a",
        "*.mp4",
        "*.flac",
        "*.ogg",
        "*.aac",
        "*.webm",
        "*.mkv",
        "*.mov",
        "*.wma",
        "*.opus",
        "*.m4v",
        "*.csv",
        "*.zip",
        "*.7z",
        "*.rar",
        "*.log",
        "*.tmp",
        "*.temp",
        "*.bak",
        "*.swp"
    ],
    "max_text_file_mb": 4
}


def load_settings(path: Path | None) -> dict:
    settings = dict(DEFAULT_SETTINGS)

    if path and path.is_file():
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
        settings.update(data)

    return settings


def should_skip_dir(name: str, settings: dict) -> bool:
    if name in set(settings.get("exclude_dirs", [])):
        return True

    for pattern in settings.get("exclude_dir_globs", []):
        if fnmatch.fnmatch(name, pattern):
            return True

    return False


def should_include_file(path: Path, settings: dict) -> bool:
    name = path.name
    lower_name = name.lower()

    for pattern in settings.get("exclude_file_globs", []):
        if fnmatch.fnmatch(lower_name, pattern.lower()):
            return False

    include_names = {
        x.lower()
        for x in settings.get("include_filenames", [])
    }
    if lower_name in include_names:
        return True

    include_exts = {
        x.lower()
        for x in settings.get("include_extensions", [])
    }

    return path.suffix.lower() in include_exts


def collect_files(root: Path, settings: dict) -> list[Path]:
    result: list[Path] = []

    for current, dirs, files in os.walk(root):
        current_path = Path(current)

        dirs[:] = [
            d for d in dirs
            if not should_skip_dir(d, settings)
        ]

        for filename in files:
            path = current_path / filename

            if should_include_file(path, settings):
                result.append(path)

    return sorted(
        result,
        key=lambda p: str(
            p.relative_to(root)
        ).lower(),
    )


def resolve_output_dir(root: Path, settings: dict) -> Path:
    mode = settings.get("output_mode", "sibling")
    dirname = settings.get(
        "output_dir_name",
        "_project_snapshots",
    )

    if mode == "inside":
        return root / dirname

    if mode == "custom":
        custom = settings.get("custom_output_dir")
        if not custom:
            raise ValueError(
                "output_mode=custom requires custom_output_dir"
            )
        return Path(custom).expanduser().resolve()

    # Default: outside Git repository.
    return root.parent / dirname / root.name


def safe_read_text(path: Path, max_bytes: int) -> tuple[str, str | None]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return "", f"stat failed: {exc}"

    if size > max_bytes:
        return "", f"skipped text dump: {size:,} bytes"

    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    )

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), None
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return "", f"read failed: {exc}"

    return "", "could not decode as text"


def build_tree(paths: list[Path], root: Path) -> str:
    tree: dict = {}

    for path in paths:
        parts = path.relative_to(root).parts
        node = tree

        for part in parts[:-1]:
            node = node.setdefault(part, {})

        node.setdefault("__files__", []).append(parts[-1])

    lines = [root.name]

    def render(node: dict, prefix: str = "") -> None:
        dirs = sorted(
            [k for k in node if k != "__files__"],
            key=str.lower,
        )
        files = sorted(
            node.get("__files__", []),
            key=str.lower,
        )

        items = [("dir", d) for d in dirs] + [
            ("file", f) for f in files
        ]

        for index, (kind, name) in enumerate(items):
            last = index == len(items) - 1
            branch = "└─ " if last else "├─ "
            lines.append(prefix + branch + name)

            if kind == "dir":
                extension = "   " if last else "│  "
                render(
                    node[name],
                    prefix + extension,
                )

    render(tree)
    return "\n".join(lines)


def write_snapshot(
    root: Path,
    files: list[Path],
    output_dir: Path,
    settings: dict,
) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    stem = (
        f"{root.name}_source_snapshot_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = output_dir / f"{stem}.zip"
    log_path = output_dir / f"{stem}.log.txt"
    tree_path = output_dir / f"{stem}_tree.md"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zf:
        for path in files:
            zf.write(
                path,
                path.relative_to(root),
            )

    max_mb = float(
        settings.get("max_text_file_mb", 4)
    )
    max_bytes = int(max_mb * 1024 * 1024)

    with log_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fp:
        fp.write(
            f"Project root: {root}\n"
            f"Generated   : {datetime.now().isoformat(timespec='seconds')}\n"
            f"Files       : {len(files)}\n\n"
        )

        for path in files:
            rel = path.relative_to(root)
            fp.write(
                "\n"
                + "=" * 88
                + f"\nFILE: {rel}\n"
                + "=" * 88
                + "\n"
            )

            text, error = safe_read_text(
                path,
                max_bytes,
            )

            if error:
                fp.write(
                    f"[{error}]\n"
                )
            else:
                fp.write(text)
                if text and not text.endswith("\n"):
                    fp.write("\n")

    tree = build_tree(files, root)

    tree_path.write_text(
        "# Project Source Tree\n\n"
        "```text\n"
        + tree
        + "\n```\n",
        encoding="utf-8",
    )

    return zip_path, log_path, tree_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a source-only project snapshot "
            "for Vocal Pitch Analyzer."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
    )
    parser.add_argument(
        "--settings",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    settings_path = (
        Path(args.settings).resolve()
        if args.settings
        else None
    )

    if not root.is_dir():
        print(
            f"[ERROR] Project root not found: {root}"
        )
        return 1

    settings = load_settings(
        settings_path
    )

    files = collect_files(
        root,
        settings,
    )

    output_dir = resolve_output_dir(
        root,
        settings,
    )

    zip_path, log_path, tree_path = write_snapshot(
        root,
        files,
        output_dir,
        settings,
    )

    total_size = sum(
        p.stat().st_size
        for p in files
        if p.exists()
    )

    print()
    print("[OK] Source snapshot completed")
    print(f"[INFO] Files : {len(files)}")
    print(f"[INFO] Size  : {total_size / 1024 / 1024:.2f} MB")
    print(f"[INFO] ZIP   : {zip_path}")
    print(f"[INFO] LOG   : {log_path}")
    print(f"[INFO] TREE  : {tree_path}")
    print()
    print(
        "[INFO] AI models, virtual environments, caches, "
        "media and build output were excluded."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
