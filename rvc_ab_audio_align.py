from __future__ import annotations

# V42_RVC_AB_ALIGNMENT_LOW_LATENCY_PATCH

from pathlib import Path
import json
import math
import wave

import numpy as np


def _read_pcm16_mono(
    path: str | Path,
) -> tuple[np.ndarray, int]:
    source = Path(
        path
    )

    with wave.open(
        str(
            source
        ),
        "rb",
    ) as reader:
        channels = int(
            reader.getnchannels()
        )
        width = int(
            reader.getsampwidth()
        )
        sample_rate = int(
            reader.getframerate()
        )
        frames = int(
            reader.getnframes()
        )
        payload = reader.readframes(
            frames
        )

    if width != 2:
        raise RuntimeError(
            f"PCM16 WAV only: {source}"
        )

    data = np.frombuffer(
        payload,
        dtype="<i2",
    ).astype(
        np.float32
    ) / 32768.0

    if channels > 1:
        usable = (
            data.size
            // channels
            * channels
        )
        data = data[
            :usable
        ].reshape(
            -1,
            channels,
        ).mean(
            axis=1,
            dtype=np.float32,
        )

    return (
        np.asarray(
            data,
            dtype=np.float32,
        ),
        max(
            8000,
            sample_rate,
        ),
    )


def _write_pcm16_mono(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    target = Path(
        path
    )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.clip(
        np.asarray(
            samples,
            dtype=np.float32,
        ).reshape(
            -1
        ),
        -1.0,
        0.9999695,
    )
    pcm = np.round(
        data
        * 32767.0
    ).astype(
        "<i2"
    )

    with wave.open(
        str(
            target
        ),
        "wb",
    ) as writer:
        writer.setnchannels(
            1
        )
        writer.setsampwidth(
            2
        )
        writer.setframerate(
            int(
                sample_rate
            )
        )
        writer.writeframes(
            pcm.tobytes()
        )


def _rms_envelope_db(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    data = np.asarray(
        samples,
        dtype=np.float32,
    ).reshape(
        -1
    )

    frame = max(
        1,
        int(
            round(
                sample_rate
                * frame_ms
                / 1000.0
            )
        ),
    )
    hop = max(
        1,
        int(
            round(
                sample_rate
                * hop_ms
                / 1000.0
            )
        ),
    )

    if data.size < frame:
        return np.zeros(
            0,
            dtype=np.float32,
        )

    square = np.square(
        data.astype(
            np.float64
        )
    )
    prefix = np.concatenate(
        (
            np.zeros(
                1,
                dtype=np.float64,
            ),
            np.cumsum(
                square
            ),
        )
    )

    starts = np.arange(
        0,
        data.size
        - frame
        + 1,
        hop,
        dtype=np.int64,
    )
    ends = starts + frame
    sums = (
        prefix[
            ends
        ]
        - prefix[
            starts
        ]
    )
    rms = np.sqrt(
        np.maximum(
            sums
            / float(
                frame
            ),
            1e-14,
        )
    )

    db = 20.0 * np.log10(
        np.maximum(
            rms,
            1e-7,
        )
    )

    # Compress silence and limiter peaks. We care about speech envelope timing,
    # not exact gain differences between ORIGINAL / RVC / Broadcast.
    return np.clip(
        db,
        -65.0,
        -6.0,
    ).astype(
        np.float32
    )


def estimate_positive_lag_ms(
    reference_path: str | Path,
    target_path: str | Path,
    *,
    max_lag_ms: int = 1500,
    hop_ms: int = 10,
) -> dict:
    ref, ref_sr = _read_pcm16_mono(
        reference_path
    )
    target, target_sr = _read_pcm16_mono(
        target_path
    )

    ref_env = _rms_envelope_db(
        ref,
        ref_sr,
        hop_ms=float(
            hop_ms
        ),
    )
    target_env = _rms_envelope_db(
        target,
        target_sr,
        hop_ms=float(
            hop_ms
        ),
    )

    if (
        ref_env.size < 20
        or target_env.size < 20
    ):
        return {
            "lag_ms": 0.0,
            "correlation": 0.0,
        }

    max_lag_frames = max(
        0,
        int(
            round(
                max_lag_ms
                / float(
                    hop_ms
                )
            )
        ),
    )

    best_correlation = -1.0
    best_lag = 0

    for lag in range(
        max_lag_frames
        + 1
    ):
        overlap = min(
            ref_env.size,
            target_env.size
            - lag,
        )

        if overlap < 100:
            break

        a = ref_env[
            :overlap
        ].astype(
            np.float64,
            copy=False,
        )
        b = target_env[
            lag:
            lag
            + overlap
        ].astype(
            np.float64,
            copy=False,
        )

        a_std = float(
            a.std()
        )
        b_std = float(
            b.std()
        )

        if (
            a_std < 1e-5
            or b_std < 1e-5
        ):
            continue

        corr = float(
            np.mean(
                (
                    a
                    - a.mean()
                )
                * (
                    b
                    - b.mean()
                )
            )
            / (
                a_std
                * b_std
            )
        )

        if corr > best_correlation:
            best_correlation = corr
            best_lag = lag

    return {
        "lag_ms": float(
            best_lag
            * hop_ms
        ),
        "correlation": float(
            best_correlation
            if best_correlation > -1.0
            else 0.0
        ),
    }


def align_wav_to_reference(
    *,
    reference_path: str | Path,
    target_path: str | Path,
    max_lag_ms: int = 1500,
    minimum_correlation: float = 0.35,
    fade_in_ms: float = 5.0,
) -> dict:
    reference = Path(
        reference_path
    )
    target = Path(
        target_path
    )

    estimate = estimate_positive_lag_ms(
        reference,
        target,
        max_lag_ms=max_lag_ms,
    )

    lag_ms = float(
        estimate[
            "lag_ms"
        ]
    )
    correlation = float(
        estimate[
            "correlation"
        ]
    )

    result = {
        "reference": str(
            reference
        ),
        "target": str(
            target
        ),
        "lag_ms": lag_ms,
        "correlation": correlation,
        "applied": False,
        "trimmed_samples": 0,
    }

    if correlation < float(
        minimum_correlation
    ):
        return result

    reference_audio, reference_sr = _read_pcm16_mono(
        reference
    )
    target_audio, target_sr = _read_pcm16_mono(
        target
    )

    trim_samples = max(
        0,
        int(
            round(
                lag_ms
                * target_sr
                / 1000.0
            )
        ),
    )

    if trim_samples > 0:
        target_audio = target_audio[
            min(
                trim_samples,
                target_audio.size,
            ):
        ]

    reference_duration = (
        reference_audio.size
        / float(
            reference_sr
        )
    )
    target_length = max(
        0,
        int(
            round(
                reference_duration
                * target_sr
            )
        ),
    )

    if target_audio.size < target_length:
        target_audio = np.pad(
            target_audio,
            (
                0,
                target_length
                - target_audio.size,
            ),
        )
    elif target_audio.size > target_length:
        target_audio = target_audio[
            :target_length
        ]

    fade_samples = min(
        target_audio.size,
        max(
            0,
            int(
                round(
                    fade_in_ms
                    * target_sr
                    / 1000.0
                )
            ),
        ),
    )

    if fade_samples > 1:
        target_audio[
            :fade_samples
        ] *= np.linspace(
            0.0,
            1.0,
            fade_samples,
            dtype=np.float32,
        )

    _write_pcm16_mono(
        target,
        target_audio,
        target_sr,
    )

    result[
        "applied"
    ] = True
    result[
        "trimmed_samples"
    ] = int(
        trim_samples
    )

    return result


def write_ab_report(
    *,
    path: str | Path,
    payload: dict,
) -> Path:
    target = Path(
        path
    )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target
