from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(slots=True)
class SubtitleGroup:
    start: float
    end: float
    segments: list


def _segment_label(segment, display_mode: str) -> str:
    if display_mode == "korean":
        return str(segment.korean_note)
    if display_mode == "both":
        return f"{segment.note} · {segment.korean_note}"
    return str(segment.note)


def build_subtitle_groups(
    segments: Sequence,
    *,
    target_seconds: float = 8.0,
    max_notes: int = 18,
    split_silence_ms: float = 400.0,
) -> list[SubtitleGroup]:
    """Group time-ordered pitch segments for readable subtitle pages.

    A group is closed before adding the next note when any of these occurs:
    - the next note would make the group exceed target_seconds
    - max_notes is already reached
    - silence between notes is >= split_silence_ms

    The subtitle itself spans from the first note start to the final note end.
    Long vocal rests therefore remain blank rather than holding stale notes.
    """
    ordered = sorted(
        [s for s in segments if float(s.end) > float(s.start)],
        key=lambda s: (float(s.start), float(s.end)),
    )

    if not ordered:
        return []

    target_seconds = max(0.1, float(target_seconds))
    max_notes = max(1, int(max_notes))
    split_silence = max(0.0, float(split_silence_ms)) / 1000.0

    groups: list[SubtitleGroup] = []
    current: list = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        groups.append(
            SubtitleGroup(
                start=float(current[0].start),
                end=float(current[-1].end),
                segments=list(current),
            )
        )
        current = []

    for segment in ordered:
        if not current:
            current.append(segment)
            continue

        previous = current[-1]
        gap = max(
            0.0,
            float(segment.start) - float(previous.end),
        )
        proposed_span = (
            float(segment.end) - float(current[0].start)
        )

        should_split = (
            len(current) >= max_notes
            or proposed_span > target_seconds
            or gap >= split_silence
        )

        if should_split:
            flush()

        current.append(segment)

    flush()
    return groups


def _wrap_tokens(
    tokens: Sequence[str],
    notes_per_line: int,
) -> list[list[str]]:
    notes_per_line = max(1, int(notes_per_line))
    return [
        list(tokens[i:i + notes_per_line])
        for i in range(0, len(tokens), notes_per_line)
    ]


def _plain_group_text(
    group: SubtitleGroup,
    *,
    display_mode: str,
    notes_per_line: int,
    newline: str,
) -> str:
    labels = [
        _segment_label(s, display_mode)
        for s in group.segments
    ]
    lines = _wrap_tokens(
        labels,
        notes_per_line,
    )
    return newline.join(
        "   ".join(line)
        for line in lines
    )


def _ass_escape(text: str) -> str:
    # Note labels normally contain no ASS control characters,
    # but escape user-visible text defensively.
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _ass_group_text(
    group: SubtitleGroup,
    *,
    display_mode: str,
    notes_per_line: int,
    active_index: int | None,
) -> str:
    labels = [
        _ass_escape(
            _segment_label(s, display_mode)
        )
        for s in group.segments
    ]

    rendered: list[str] = []

    for idx, label in enumerate(labels):
        if idx == active_index:
            # ASS uses BGR hexadecimal. This is a bright highlight;
            # the base style remains white.
            rendered.append(
                r"{\c&H00FFFF&\b1}" +
                "【" + label + "】" +
                r"{\rNoteTimeline}"
            )
        else:
            rendered.append(label)

    lines = _wrap_tokens(
        rendered,
        notes_per_line,
    )
    return r"\N".join(
        "   ".join(line)
        for line in lines
    )


def _format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(float(seconds) * 100.0)))
    cs = total_cs % 100
    total_s = total_cs // 100
    sec = total_s % 60
    total_m = total_s // 60
    minute = total_m % 60
    hour = total_m // 60
    return f"{hour}:{minute:02d}:{sec:02d}.{cs:02d}"


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    sec = total_s % 60
    total_m = total_s // 60
    minute = total_m % 60
    hour = total_m // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def _ass_intervals_for_group(
    group: SubtitleGroup,
) -> list[tuple[float, float, int | None]]:
    """Create stable display intervals for current-note highlighting.

    The whole group is shown in every interval. Only the highlighted token changes.
    During a small gap between notes, all tokens are shown unhighlighted.
    """
    points = {
        float(group.start),
        float(group.end),
    }

    for segment in group.segments:
        points.add(
            max(
                float(group.start),
                min(float(group.end), float(segment.start)),
            )
        )
        points.add(
            max(
                float(group.start),
                min(float(group.end), float(segment.end)),
            )
        )

    ordered = sorted(points)
    intervals: list[tuple[float, float, int | None]] = []

    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue

        midpoint = (start + end) * 0.5
        active_index: int | None = None

        for idx, segment in enumerate(group.segments):
            if (
                float(segment.start) <= midpoint
                < float(segment.end)
            ):
                active_index = idx
                break

        intervals.append(
            (start, end, active_index)
        )

    return intervals


def write_ass(
    path: str | Path,
    groups: Sequence[SubtitleGroup],
    *,
    display_mode: str = "korean",
    notes_per_line: int = 6,
    highlight_current: bool = True,
    font_size: int = 44,
) -> Path:
    path = Path(path)

    header = f"""[Script Info]
; Generated by Vocal Pitch Analyzer
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: NoteTimeline,Arial,{int(font_size)},&H00FFFFFF,&H0000FFFF,&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = [header.rstrip("\n")]

    for group in groups:
        if highlight_current:
            intervals = _ass_intervals_for_group(group)

            for start, end, active_index in intervals:
                text = _ass_group_text(
                    group,
                    display_mode=display_mode,
                    notes_per_line=notes_per_line,
                    active_index=active_index,
                )
                lines.append(
                    "Dialogue: 0,"
                    f"{_format_ass_time(start)},"
                    f"{_format_ass_time(end)},"
                    "NoteTimeline,,0,0,0,,"
                    f"{text}"
                )
        else:
            text = _ass_group_text(
                group,
                display_mode=display_mode,
                notes_per_line=notes_per_line,
                active_index=None,
            )
            lines.append(
                "Dialogue: 0,"
                f"{_format_ass_time(group.start)},"
                f"{_format_ass_time(group.end)},"
                "NoteTimeline,,0,0,0,,"
                f"{text}"
            )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig",
    )
    return path


def write_srt(
    path: str | Path,
    groups: Sequence[SubtitleGroup],
    *,
    display_mode: str = "korean",
    notes_per_line: int = 6,
) -> Path:
    path = Path(path)
    blocks: list[str] = []

    for idx, group in enumerate(groups, start=1):
        text = _plain_group_text(
            group,
            display_mode=display_mode,
            notes_per_line=notes_per_line,
            newline="\n",
        )
        blocks.append(
            f"{idx}\n"
            f"{_format_srt_time(group.start)} --> {_format_srt_time(group.end)}\n"
            f"{text}"
        )

    path.write_text(
        "\n\n".join(blocks) + ("\n" if blocks else ""),
        encoding="utf-8-sig",
    )
    return path
