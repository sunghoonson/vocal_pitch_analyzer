# Vocal Pitch Analyzer - Project Dev Tools

Target project:

```text
C:\dev\vocal_pitch_prototype_v1
```

Copy this package into the project root so the layout becomes:

```text
C:\dev\vocal_pitch_prototype_v1
├─ collect_project_sources.bat
├─ commit_push_datetime.bat
└─ dev_tools
   ├─ collect_project_sources.py
   └─ collect_project_sources_settings.json
```

## 1. collect_project_sources.bat

Creates a source-only snapshot of the project.

Output is deliberately placed outside the Git repository:

```text
C:\dev\_project_snapshots\vocal_pitch_prototype_v1\
```

Each run creates:

```text
vocal_pitch_prototype_v1_source_snapshot_YYYYMMDD_HHMMSS.zip
vocal_pitch_prototype_v1_source_snapshot_YYYYMMDD_HHMMSS.log.txt
vocal_pitch_prototype_v1_source_snapshot_YYYYMMDD_HHMMSS_tree.md
```

Excluded by default:

```text
.git
.venv
.venv_separator
cache
models
logs
build
dist
backup_before_*
backup_*

AI model files
audio/video files
generated CSV
ZIP/7z archives
EXE/DLL
```

The settings are editable here:

```text
dev_tools\collect_project_sources_settings.json
```

## 2. commit_push_datetime.bat

Runs:

```text
git status --short
git add -A
git commit
git push
```

Commit title is automatically:

```text
yyyy-MM-dd HH:mm:ss
```

An optional one-line commit body can be entered in the console.

For this repository it also repairs the earlier remote-name typo automatically:

```text
origine -> origin
```

If `origin` is completely missing, it adds:

```text
https://github.com/sunghoonson/vocal_pitch_analyzer.git
```

The currently checked-out branch is pushed, so it remains usable if you later work on
a branch other than `main`.
