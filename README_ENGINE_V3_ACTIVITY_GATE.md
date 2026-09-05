# Vocal Pitch Analyzer v1.4 / Pitch Engine v3 Activity Gate

Target:

```text
C:\dev\vocal_pitch_prototype_v1
```

## Why this patch exists

After successful BS-RoFormer vocal separation, pYIN still detects pitch in very weak
residual/reverb regions where nobody is actually singing.

In the test `vocals.wav`, false notes near the intro/outro were often tens of dB
quieter than actual sung sections.

Example pattern:

```text
real singing       roughly -10 ~ -30 dBFS
weak residuals     roughly -50 ~ -90 dBFS
```

A high pYIN probability by itself does NOT mean "this is definitely the lead vocal".
It means the remaining signal looks periodic.

## Engine v3 pipeline

```text
vocals.wav
    ↓
Raw pYIN F0
    +
RMS energy envelope
    ↓
Adaptive vocal-activity gate
    ↓
pYIN confidence gate
    ↓
short pitch dropout bridge
    ↓
pitch smoothing
    ↓
note segmentation
```

## Adaptive energy threshold

Default:

```text
reference = 90th percentile RMS level
threshold = max(-55 dBFS, reference - 32 dB)
```

So the threshold adapts to differently mastered songs.

For the previously supplied vocal stem, this is expected to land around the mid
-40 dBFS range, which removes most intro/outro residual notes while retaining the
normal singing sections.

## Other safeguards

### Gate hysteresis

Default:

```text
4 dB
```

Once singing starts, the gate does not rapidly open/close around one threshold.

### Short activity gap fill

Default:

```text
100 ms
```

Tiny dips caused by consonants/breaths are bridged.

### Minimum activity island

Default:

```text
80 ms
```

Very short residual-energy spikes are discarded.

### Range calculation

Displayed highest/lowest note now requires, by default:

```text
duration >= 100 ms
confidence >= 0.35
```

So a single tiny E6/C6 glitch is not automatically reported as the song's highest
note.

## GUI options

A new `보컬 활동 게이트 - Engine v3` section is added.

Recommended first test: leave all defaults unchanged.

## Raw CSV v1.4

New diagnostic columns:

```text
rms_dbfs
energy_gate_active
energy_threshold_dbfs
accepted_by_all_gates
```

These make it possible to see exactly why a frame was kept or rejected.

## Apply

Run:

```text
APPLY_PATCH.bat
```

The patch backs up current `main.py` and `pitch_analyzer.py`.

It does NOT change:

- CUDA
- PyTorch
- audio-separator
- BS-RoFormer model/cache
- vocal_separator.py

The already separated vocals cache can therefore be reused.
