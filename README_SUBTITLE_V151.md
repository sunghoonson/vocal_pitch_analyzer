# Vocal Pitch Analyzer v1.5.1 - Korean + Bracket Subtitle UI

Target:

```text
C:\dev\vocal_pitch_prototype_v1
```

This is a small readability patch on top of v1.5 grouped note subtitles.

## Changes

### 1. Korean note names are now the default

Old default:

```text
A3  A#3  B3  C#4  C4  ...
```

New default:

```text
1옥타브 라
1옥타브 라♯
1옥타브 시
2옥타브 도♯
...
```

The GUI still lets you switch back to:

```text
D4
D4 · 2옥타브 레
```

### 2. Current note uses visible brackets

ASS current-note display changes from color-only:

```text
2옥타브 미   2옥타브 파♯   2옥타브 레
```

to:

```text
2옥타브 미   【2옥타브 파♯】   2옥타브 레
```

The bracketed note is also highlighted in color and bold.

This makes the active note easy to locate even when the player or subtitle renderer
changes colors slightly.

### 3. Fewer notes per line by default

Old:

```text
9 notes / line
```

New:

```text
6 notes / line
```

Korean labels are much longer, so this prevents the subtitle from becoming excessively
wide.

### 4. Smaller ASS font default

Old:

```text
54
```

New:

```text
44
```

This better fits two lines of full Korean octave names on 1080p video.

## Recommended defaults

```text
Format            ASS
Display           2옥타브 레 (한글, 권장)
Max time          8 sec
Max notes         18
Long silence      400 ms
Notes per line    6
Current highlight ON
```

## Apply

Run:

```text
APPLY_PATCH.bat
```

The existing `main.py` and `subtitle_generator.py` are backed up first.

No changes are made to pitch analysis, CUDA, BS-RoFormer, or vocal cache.
