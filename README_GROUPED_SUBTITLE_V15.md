# Vocal Pitch Analyzer v1.5 - Grouped Note Subtitle GUI

Target project:

```text
C:\dev\vocal_pitch_prototype_v1
```

This patch keeps the existing v1.4 pipeline:

```text
BS-RoFormer
→ Vocal Activity Gate
→ pYIN / Pitch Engine v3
→ note segments
```

and adds:

```text
note segments
→ grouped subtitle timeline
→ ASS / SRT
```

## Default grouping

```text
Maximum group time     8 seconds
Maximum notes          18
Split on long silence  400 ms
Notes per line         9
```

A group is closed before the next note when the first of these conditions is reached:

- adding the next note would exceed 8 seconds
- the group already contains 18 notes
- silence before the next note is 400 ms or longer

This means the grouping is not rigidly fixed to exactly 8-second blocks.
Dense passages split by note count, while sparse passages can use the available time.

## ASS recommended

ASS displays the whole group at a stable position.

Example:

```text
D4   E4   F#4   G4   A4   G4   F#4   E4
D4   E4   F#4   A4   G4
```

With `현재 음 강조` enabled, the whole group remains visible while the note that is
currently active on the original timeline is highlighted.

The subtitle event timing still comes from the original `start_sec` / `end_sec`
pitch segments, so the highlight follows the source video timeline.

## SRT

SRT contains one cue per group.

It cannot perform per-note live highlighting, but it is highly compatible with media
players.

## Display modes

```text
D4
2옥타브 레
D4 · 2옥타브 레
```

## GUI controls

A compact `음계 자막 생성 - v1.5` row is added with:

- format: ASS / SRT / ASS + SRT
- display format
- maximum group time
- maximum notes per group
- long-silence split threshold
- notes per line
- current-note highlight
- Generate button

## Important

The subtitle generator uses the current in-memory analysis result directly.

You do NOT need to save and re-open the CSV before generating a subtitle.

If you already have a separated `vocals.wav`, you may select that WAV and choose
`원본 전체 믹스 분석` (meaning "analyze selected file directly") and then generate
the subtitle from that result.

## Apply

Run:

```text
APPLY_PATCH.bat
```

The current `main.py` is backed up automatically.

This patch does not change:

- `pitch_analyzer.py`
- `vocal_separator.py`
- CUDA / Torch
- BS-RoFormer
- vocal cache
