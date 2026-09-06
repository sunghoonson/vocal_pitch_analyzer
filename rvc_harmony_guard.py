from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import math

import librosa
import numpy as np
import soundfile as sf


LogCallback = Callable[[str], None]

# V26_ARTIFACT_PRIORITY_MANUAL_BYPASS_PATCH
# V25_RVC_ARTIFACT_GUARD_PATCH
GUARD_VERSION = "2.6"

SENSITIVITY_LABELS = {
    "low": "낮음",
    "medium": "보통",
    "high": "높음",
}


@dataclass(slots=True)
class HarmonyGuardReport:
    duration_seconds: float
    sensitivity: str
    crossfade_ms: int
    safe_seconds: float
    blend_seconds: float
    fallback_seconds: float
    risky_region_count: int
    f0_jump_count: int
    mean_risk: float
    max_risk: float
    fallback_gain: float
    alignment_ms: float
    risky_regions: list[tuple[float, float]]
    guard_version: str = GUARD_VERSION
    artifact_guard_enabled: bool = True
    artifact_detected_seconds: float = 0.0
    artifact_strong_seconds: float = 0.0
    artifact_region_count: int = 0
    artifact_f0_mismatch_count: int = 0
    artifact_output_jump_count: int = 0
    artifact_voicing_mismatch_seconds: float = 0.0
    mean_artifact_risk: float = 0.0
    max_artifact_risk: float = 0.0
    artifact_priority_mode: bool = True
    harmony_hint_cap: float = 0.0
    manual_bypass_seconds: float = 0.0
    manual_region_count: int = 0
    manual_regions: list[tuple[float, float]] | None = None


def _emit(
    callback: LogCallback | None,
    message: str,
) -> None:
    clean = str(message).strip()

    if clean and callback:
        callback(clean)


def _smoothstep(
    value: np.ndarray,
) -> np.ndarray:
    x = np.clip(
        value,
        0.0,
        1.0,
    )
    return (
        x
        * x
        * (
            3.0
            - 2.0 * x
        )
    )


def _moving_average(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    if (
        radius <= 0
        or values.size <= 2
    ):
        return values.astype(
            np.float32,
            copy=True,
        )

    size = (
        radius * 2
        + 1
    )
    kernel = np.full(
        size,
        1.0 / size,
        dtype=np.float32,
    )

    padded = np.pad(
        values.astype(
            np.float32,
            copy=False,
        ),
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    ).astype(
        np.float32,
        copy=False,
    )


def _moving_max(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    if (
        radius <= 0
        or values.size <= 2
    ):
        return values.astype(
            np.float32,
            copy=True,
        )

    padded = np.pad(
        values.astype(
            np.float32,
            copy=False,
        ),
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    windows = [
        padded[
            offset:
            offset
            + values.size
        ]
        for offset in range(
            radius * 2
            + 1
        )
    ]

    return np.maximum.reduce(
        windows
    ).astype(
        np.float32,
        copy=False,
    )


def _frame_risk_to_fallback(
    risk: np.ndarray,
    sensitivity: str,
) -> np.ndarray:
    sensitivity = (
        sensitivity
        if sensitivity in SENSITIVITY_LABELS
        else "medium"
    )

    thresholds = {
        "low": (
            0.72,
            0.95,
        ),
        "medium": (
            0.55,
            0.85,
        ),
        "high": (
            0.42,
            0.75,
        ),
    }

    start, full = (
        thresholds[
            sensitivity
        ]
    )

    scaled = (
        risk
        - start
    ) / max(
        1e-6,
        full - start,
    )

    return (
        _smoothstep(
            scaled
        )
        * 0.92
    ).astype(
        np.float32,
        copy=False,
    )


def _harmonic_coverage(
    power: np.ndarray,
    frequencies: np.ndarray,
    f0: np.ndarray,
    *,
    min_frequency: float = 80.0,
    max_frequency: float = 8000.0,
) -> np.ndarray:
    frame_count = min(
        power.shape[1],
        f0.size,
    )

    result = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    band = (
        (
            frequencies
            >= min_frequency
        )
        & (
            frequencies
            <= max_frequency
        )
    )

    if not np.any(
        band
    ):
        return result

    frequency_resolution = (
        frequencies[1]
        - frequencies[0]
        if frequencies.size >= 2
        else 1.0
    )

    for frame in range(
        frame_count
    ):
        fundamental = float(
            f0[
                frame
            ]
        )

        if (
            not math.isfinite(
                fundamental
            )
            or fundamental <= 0.0
        ):
            continue

        total = float(
            np.sum(
                power[
                    band,
                    frame,
                ]
            )
        )

        if total <= 1e-12:
            continue

        explained = 0.0
        harmonic = 1

        while (
            harmonic
            * fundamental
            <= max_frequency
        ):
            frequency = (
                harmonic
                * fundamental
            )
            center = int(
                round(
                    frequency
                    / frequency_resolution
                )
            )
            half_width_hz = max(
                15.0,
                frequency * 0.012,
            )
            half_width = max(
                1,
                int(
                    round(
                        half_width_hz
                        / frequency_resolution
                    )
                ),
            )
            lo = max(
                0,
                center
                - half_width,
            )
            hi = min(
                power.shape[0],
                center
                + half_width
                + 1,
            )

            explained += float(
                np.sum(
                    power[
                        lo:hi,
                        frame,
                    ]
                )
            )
            harmonic += 1

        result[
            frame
        ] = min(
            1.0,
            explained
            / total,
        )

    return result


def _isolated_f0_jump_risk(
    f0: np.ndarray,
) -> tuple[np.ndarray, int]:
    risk = np.zeros(
        f0.size,
        dtype=np.float32,
    )

    valid = np.isfinite(
        f0
    ) & (
        f0 > 0.0
    )

    midi = np.full(
        f0.size,
        np.nan,
        dtype=np.float32,
    )

    midi[
        valid
    ] = (
        69.0
        + 12.0
        * np.log2(
            f0[
                valid
            ]
            / 440.0
        )
    )

    jump_count = 0

    for index in range(
        1,
        max(
            1,
            f0.size - 1,
        ),
    ):
        if (
            index + 1
            >= f0.size
            or not np.isfinite(
                midi[
                    index - 1:
                    index + 2
                ]
            ).all()
        ):
            continue

        before = float(
            midi[
                index - 1
            ]
        )
        current = float(
            midi[
                index
            ]
        )
        after = float(
            midi[
                index + 1
            ]
        )

        jump_in = abs(
            current
            - before
        )
        returns = abs(
            after
            - before
        )

        if (
            jump_in >= 5.0
            and returns <= 2.0
        ):
            severity = min(
                1.0,
                (
                    jump_in
                    - 4.0
                )
                / 8.0,
            )
            risk[
                index
            ] = max(
                risk[
                    index
                ],
                0.65
                + 0.35
                * severity,
            )
            jump_count += 1

    valid_midi = np.nan_to_num(
        midi,
        nan=0.0,
    )

    local_change = np.abs(
        np.diff(
            valid_midi,
            prepend=valid_midi[
                :1
            ],
        )
    )

    unstable = (
        local_change
        >= 7.0
    ) & valid

    risk[
        unstable
    ] = np.maximum(
        risk[
            unstable
        ],
        0.55,
    )

    return (
        risk,
        jump_count,
    )


def _regions_from_mask(
    mask: np.ndarray,
    frame_seconds: float,
    *,
    threshold: float = 0.35,
    min_seconds: float = 0.12,
) -> list[tuple[float, float]]:
    active = (
        mask
        >= threshold
    )

    regions: list[
        tuple[
            float,
            float,
        ]
    ] = []

    start: int | None = None

    for index, value in enumerate(
        active
    ):
        if value and start is None:
            start = index
        elif (
            not value
            and start is not None
        ):
            end = index

            if (
                end
                - start
            ) * frame_seconds >= min_seconds:
                regions.append(
                    (
                        start
                        * frame_seconds,
                        end
                        * frame_seconds,
                    )
                )

            start = None

    if start is not None:
        end = active.size

        if (
            end
            - start
        ) * frame_seconds >= min_seconds:
            regions.append(
                (
                    start
                    * frame_seconds,
                    end
                    * frame_seconds,
                )
            )

    return regions



def _mono_audio(
    data: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        data,
        dtype=np.float32,
    )

    if array.ndim == 1:
        return array

    if array.ndim != 2:
        return np.reshape(
            array,
            (-1,),
        ).astype(
            np.float32,
            copy=False,
        )

    return np.mean(
        array,
        axis=1,
        dtype=np.float32,
    )


def _analysis_resample(
    mono: np.ndarray,
    original_sr: int,
    analysis_sr: int,
) -> np.ndarray:
    if int(
        original_sr
    ) == int(
        analysis_sr
    ):
        return mono.astype(
            np.float32,
            copy=False,
        )

    return librosa.resample(
        mono.astype(
            np.float32,
            copy=False,
        ),
        orig_sr=int(
            original_sr
        ),
        target_sr=int(
            analysis_sr
        ),
        res_type="soxr_hq",
    ).astype(
        np.float32,
        copy=False,
    )


def _safe_peak_normalize(
    values: np.ndarray,
) -> np.ndarray:
    peak = float(
        np.max(
            np.abs(
                values
            )
        )
    ) if values.size else 0.0

    if peak <= 1e-8:
        return values.astype(
            np.float32,
            copy=True,
        )

    return (
        values
        / peak
    ).astype(
        np.float32,
        copy=False,
    )


def _midi_from_f0(
    f0: np.ndarray,
) -> np.ndarray:
    result = np.full(
        f0.size,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(
        f0
    ) & (
        f0 > 0.0
    )

    result[
        valid
    ] = (
        69.0
        + 12.0
        * np.log2(
            f0[
                valid
            ]
            / 440.0
        )
    )

    return result


def _differential_output_jump_risk(
    reference_f0: np.ndarray,
    rvc_f0: np.ndarray,
) -> tuple[np.ndarray, int]:
    count = min(
        reference_f0.size,
        rvc_f0.size,
    )
    risk = np.zeros(
        count,
        dtype=np.float32,
    )

    ref_midi = _midi_from_f0(
        reference_f0[
            :count
        ]
    )
    out_midi = _midi_from_f0(
        rvc_f0[
            :count
        ]
    )

    jump_count = 0

    for index in range(
        1,
        max(
            1,
            count - 1,
        ),
    ):
        if (
            index + 1 >= count
            or not np.isfinite(
                ref_midi[
                    index - 1:
                    index + 2
                ]
            ).all()
            or not np.isfinite(
                out_midi[
                    index - 1:
                    index + 2
                ]
            ).all()
        ):
            continue

        ref_before = float(
            ref_midi[
                index - 1
            ]
        )
        ref_current = float(
            ref_midi[
                index
            ]
        )
        ref_after = float(
            ref_midi[
                index + 1
            ]
        )

        out_before = float(
            out_midi[
                index - 1
            ]
        )
        out_current = float(
            out_midi[
                index
            ]
        )
        out_after = float(
            out_midi[
                index + 1
            ]
        )

        reference_change = max(
            abs(
                ref_current
                - ref_before
            ),
            abs(
                ref_after
                - ref_current
            ),
        )

        output_jump = abs(
            out_current
            - out_before
        )
        output_returns = abs(
            out_after
            - out_before
        )

        # Directly target "RVC alone jumped to a wrong note and returned".
        if (
            output_jump >= 4.0
            and reference_change <= 2.5
            and output_returns <= 2.5
        ):
            severity = np.clip(
                (
                    output_jump
                    - 3.0
                )
                / 8.0,
                0.0,
                1.0,
            )

            risk[
                index
            ] = max(
                risk[
                    index
                ],
                float(
                    0.80
                    + 0.20
                    * severity
                ),
            )
            jump_count += 1

    return (
        risk,
        jump_count,
    )


def _chroma_similarity(
    reference_power: np.ndarray,
    rvc_power: np.ndarray,
    sr: int,
    *,
    n_fft: int,
) -> np.ndarray:
    reference_chroma = (
        librosa.feature.chroma_stft(
            S=reference_power,
            sr=int(
                sr
            ),
            n_fft=int(
                n_fft
            ),
        )
    )
    rvc_chroma = (
        librosa.feature.chroma_stft(
            S=rvc_power,
            sr=int(
                sr
            ),
            n_fft=int(
                n_fft
            ),
        )
    )

    count = min(
        reference_chroma.shape[1],
        rvc_chroma.shape[1],
    )

    if count <= 0:
        return np.zeros(
            0,
            dtype=np.float32,
        )

    reference_chroma = reference_chroma[
        :,
        :count,
    ].astype(
        np.float32,
        copy=False,
    )
    rvc_chroma = rvc_chroma[
        :,
        :count,
    ].astype(
        np.float32,
        copy=False,
    )

    reference_norm = np.linalg.norm(
        reference_chroma,
        axis=0,
    )
    rvc_norm = np.linalg.norm(
        rvc_chroma,
        axis=0,
    )

    denominator = np.maximum(
        reference_norm
        * rvc_norm,
        1e-8,
    )

    similarity = np.sum(
        reference_chroma
        * rvc_chroma,
        axis=0,
    ) / denominator

    return np.clip(
        similarity,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )


def _artifact_risk_to_fallback(
    risk: np.ndarray,
    sensitivity: str,
) -> np.ndarray:
    sensitivity = (
        sensitivity
        if sensitivity in SENSITIVITY_LABELS
        else "medium"
    )

    # Artifact Guard intentionally reacts earlier than the input-only
    # Harmony Guard because it compares the already-generated RVC output
    # against a same-key reference.
    thresholds = {
        "low": (
            0.58,
            0.88,
        ),
        "medium": (
            0.36,
            0.72,
        ),
        "high": (
            0.24,
            0.58,
        ),
    }

    start, full = (
        thresholds[
            sensitivity
        ]
    )

    scaled = (
        risk
        - start
    ) / max(
        1e-6,
        full - start,
    )

    return (
        _smoothstep(
            scaled
        )
        * 0.98
    ).astype(
        np.float32,
        copy=False,
    )


def analyze_rvc_artifact_risk(
    rvc_data: np.ndarray,
    reference_data: np.ndarray,
    sample_rate: int,
    *,
    sensitivity: str = "medium",
    crossfade_ms: int = 500,
    analysis_sr: int = 22050,
    log_callback: LogCallback | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    sensitivity = (
        sensitivity
        if sensitivity in SENSITIVITY_LABELS
        else "medium"
    )

    _emit(
        log_callback,
        (
            "[Artifact Guard] 2차 RVC 출력 검증 시작 "
            f"(민감도={SENSITIVITY_LABELS[sensitivity]})"
        ),
    )

    rvc_mono = _analysis_resample(
        _mono_audio(
            rvc_data
        ),
        int(
            sample_rate
        ),
        int(
            analysis_sr
        ),
    )
    reference_mono = _analysis_resample(
        _mono_audio(
            reference_data
        ),
        int(
            sample_rate
        ),
        int(
            analysis_sr
        ),
    )

    target_length = min(
        rvc_mono.size,
        reference_mono.size,
    )

    rvc_mono = rvc_mono[
        :target_length
    ]
    reference_mono = reference_mono[
        :target_length
    ]

    rvc_mono = _safe_peak_normalize(
        rvc_mono
    )
    reference_mono = _safe_peak_normalize(
        reference_mono
    )

    frame_length = 2048
    hop_length = 512
    n_fft = 2048

    reference_f0, _, reference_probability = (
        librosa.pyin(
            reference_mono,
            fmin=65.0,
            fmax=1600.0,
            sr=int(
                analysis_sr
            ),
            frame_length=frame_length,
            hop_length=hop_length,
        )
    )
    rvc_f0, _, rvc_probability = (
        librosa.pyin(
            rvc_mono,
            fmin=65.0,
            fmax=1600.0,
            sr=int(
                analysis_sr
            ),
            frame_length=frame_length,
            hop_length=hop_length,
        )
    )

    reference_magnitude = np.abs(
        librosa.stft(
            reference_mono,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=True,
        )
    ).astype(
        np.float32,
        copy=False,
    )
    rvc_magnitude = np.abs(
        librosa.stft(
            rvc_mono,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=True,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    reference_power = (
        reference_magnitude
        * reference_magnitude
    )
    rvc_power = (
        rvc_magnitude
        * rvc_magnitude
    )

    reference_rms = librosa.feature.rms(
        y=reference_mono,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[
        0
    ]

    chroma_similarity = _chroma_similarity(
        reference_power,
        rvc_power,
        int(
            analysis_sr
        ),
        n_fft=n_fft,
    )

    count = min(
        reference_f0.size,
        rvc_f0.size,
        reference_probability.size,
        rvc_probability.size,
        reference_power.shape[1],
        rvc_power.shape[1],
        reference_rms.size,
        chroma_similarity.size,
    )

    reference_f0 = reference_f0[
        :count
    ]
    rvc_f0 = rvc_f0[
        :count
    ]
    reference_probability = np.nan_to_num(
        reference_probability[
            :count
        ],
        nan=0.0,
    ).astype(
        np.float32,
        copy=False,
    )
    rvc_probability = np.nan_to_num(
        rvc_probability[
            :count
        ],
        nan=0.0,
    ).astype(
        np.float32,
        copy=False,
    )
    reference_power = reference_power[
        :,
        :count,
    ]
    rvc_power = rvc_power[
        :,
        :count,
    ]
    reference_rms = reference_rms[
        :count
    ].astype(
        np.float32,
        copy=False,
    )
    chroma_similarity = chroma_similarity[
        :count
    ]

    reference_rms_db = (
        20.0
        * np.log10(
            np.maximum(
                reference_rms,
                1e-8,
            )
        )
    )
    reference_db = float(
        np.percentile(
            reference_rms_db,
            95,
        )
    )
    active = (
        reference_rms_db
        >= (
            reference_db
            - 42.0
        )
    )

    reference_midi = _midi_from_f0(
        reference_f0
    )
    rvc_midi = _midi_from_f0(
        rvc_f0
    )

    both_valid = (
        np.isfinite(
            reference_midi
        )
        & np.isfinite(
            rvc_midi
        )
        & active
    )

    midi_error = np.zeros(
        count,
        dtype=np.float32,
    )
    midi_error[
        both_valid
    ] = np.abs(
        rvc_midi[
            both_valid
        ]
        - reference_midi[
            both_valid
        ]
    )

    pitch_risk = np.clip(
        (
            midi_error
            - 0.65
        )
        / 4.35,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    # A same-key Pitch-only reference is confidently voiced while RVC has
    # lost the voice or pyIN confidence collapsed.
    reference_confident = (
        reference_probability
        >= 0.62
    ) & active

    rvc_missing = (
        ~np.isfinite(
            rvc_f0
        )
    ) | (
        rvc_probability
        < 0.20
    )

    voicing_mismatch = (
        reference_confident
        & rvc_missing
    )

    confidence_drop_risk = np.clip(
        (
            reference_probability
            - rvc_probability
            - 0.12
        )
        / 0.50,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    frequencies = librosa.fft_frequencies(
        sr=int(
            analysis_sr
        ),
        n_fft=n_fft,
    )
    rvc_coverage = _harmonic_coverage(
        rvc_power,
        frequencies,
        rvc_f0,
    )
    coverage_risk = np.clip(
        (
            0.80
            - rvc_coverage
        )
        / 0.48,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    chroma_risk = np.clip(
        (
            0.82
            - chroma_similarity
        )
        / 0.42,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    jump_risk, output_jump_count = (
        _differential_output_jump_risk(
            reference_f0,
            rvc_f0,
        )
    )

    artifact_risk = (
        pitch_risk
        * 0.54
        + confidence_drop_risk
        * 0.16
        + coverage_risk
        * 0.14
        + chroma_risk
        * 0.16
    )

    artifact_risk = np.maximum(
        artifact_risk,
        jump_risk,
    )

    artifact_risk[
        voicing_mismatch
    ] = np.maximum(
        artifact_risk[
            voicing_mismatch
        ],
        0.94,
    )

    artifact_risk[
        ~active
    ] = 0.0

    frame_seconds = (
        hop_length
        / float(
            analysis_sr
        )
    )

    # Expand around an actually-detected output failure, then smooth the
    # transition.  This keeps short "삑사리" events from being averaged away.
    dilation_radius = max(
        1,
        int(
            round(
                min(
                    0.18,
                    max(
                        0.06,
                        crossfade_ms
                        / 1000.0
                        * 0.30,
                    ),
                )
                / frame_seconds
            )
        ),
    )
    artifact_risk = _moving_max(
        artifact_risk,
        dilation_radius,
    )

    fallback = _artifact_risk_to_fallback(
        artifact_risk,
        sensitivity,
    )

    transition_radius = max(
        1,
        int(
            round(
                min(
                    0.20,
                    max(
                        0.04,
                        crossfade_ms
                        / 1000.0
                        * 0.22,
                    ),
                )
                / frame_seconds
            )
        ),
    )
    fallback = _moving_max(
        fallback,
        max(
            1,
            transition_radius // 2,
        ),
    )
    fallback = _moving_average(
        fallback,
        transition_radius,
    )
    fallback = np.clip(
        fallback,
        0.0,
        0.98,
    ).astype(
        np.float32,
        copy=False,
    )

    times = (
        np.arange(
            count,
            dtype=np.float32,
        )
        * frame_seconds
    )

    mismatch_mask = (
        midi_error
        >= 1.5
    ) & both_valid

    mismatch_count = int(
        np.sum(
            mismatch_mask
        )
    )

    voicing_mismatch_seconds = float(
        np.sum(
            voicing_mismatch
        )
        * frame_seconds
    )

    detected_seconds = float(
        np.sum(
            fallback
            >= 0.15
        )
        * frame_seconds
    )
    strong_seconds = float(
        np.sum(
            fallback
            >= 0.65
        )
        * frame_seconds
    )

    regions = _regions_from_mask(
        fallback,
        frame_seconds,
        threshold=0.35,
        min_seconds=0.08,
    )

    metrics = {
        "artifact_detected_seconds": detected_seconds,
        "artifact_strong_seconds": strong_seconds,
        "artifact_region_count": len(
            regions
        ),
        "artifact_f0_mismatch_count": mismatch_count,
        "artifact_output_jump_count": int(
            output_jump_count
        ),
        "artifact_voicing_mismatch_seconds": voicing_mismatch_seconds,
        "mean_artifact_risk": float(
            np.mean(
                artifact_risk
            )
        ) if artifact_risk.size else 0.0,
        "max_artifact_risk": float(
            np.max(
                artifact_risk
            )
        ) if artifact_risk.size else 0.0,
    }

    _emit(
        log_callback,
        (
            "[Artifact Guard] 출력 검증 완료: "
            f"감지 {detected_seconds:.1f}s / "
            f"강한 우회 {strong_seconds:.1f}s / "
            f"구간 {len(regions)}개 / "
            f"F0 불일치 frame {mismatch_count} / "
            f"RVC 단독 급변 {output_jump_count}건 / "
            f"유성 소실 {voicing_mismatch_seconds:.1f}s"
        ),
    )

    for start, end in regions[
        :12
    ]:
        _emit(
            log_callback,
            (
                "[Artifact Guard] RVC 이상 후보 "
                f"{start:.2f}s ~ {end:.2f}s"
            ),
        )

    return (
        times,
        fallback,
        artifact_risk,
        metrics,
    )


def analyze_harmony_risk(
    vocal_path: str | Path,
    *,
    sensitivity: str = "medium",
    crossfade_ms: int = 500,
    analysis_sr: int = 22050,
    log_callback: LogCallback | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    HarmonyGuardReport,
]:
    path = Path(
        vocal_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    sensitivity = (
        sensitivity
        if sensitivity in SENSITIVITY_LABELS
        else "medium"
    )
    crossfade_ms = max(
        100,
        min(
            2000,
            int(
                crossfade_ms
            ),
        ),
    )

    _emit(
        log_callback,
        (
            "[Harmony Guard] 보컬 다성/F0 안정성 분석 시작 "
            f"(민감도={SENSITIVITY_LABELS[sensitivity]}, "
            f"crossfade={crossfade_ms}ms)"
        ),
    )

    y, sr = librosa.load(
        str(
            path
        ),
        sr=analysis_sr,
        mono=True,
    )

    if y.size <= 0:
        raise RuntimeError(
            "Harmony Guard 분석용 보컬이 비어 있습니다."
        )

    peak = float(
        np.max(
            np.abs(
                y
            )
        )
    )

    if peak > 1e-7:
        y = (
            y
            / peak
        ).astype(
            np.float32,
            copy=False,
        )

    frame_length = 2048
    hop_length = 512
    n_fft = 2048

    f0, _voiced_flag, voiced_probability = (
        librosa.pyin(
            y,
            fmin=65.0,
            fmax=1600.0,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    )

    magnitude = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=True,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    power = (
        magnitude
        * magnitude
    )

    rms = (
        librosa.feature.rms(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[
            0
        ]
    )

    count = min(
        f0.size,
        power.shape[1],
        rms.size,
        voiced_probability.size,
    )

    f0 = f0[
        :count
    ]
    voiced_probability = np.nan_to_num(
        voiced_probability[
            :count
        ],
        nan=0.0,
    ).astype(
        np.float32,
        copy=False,
    )
    power = power[
        :,
        :count,
    ]
    rms = rms[
        :count
    ].astype(
        np.float32,
        copy=False,
    )

    frequencies = (
        librosa.fft_frequencies(
            sr=sr,
            n_fft=n_fft,
        )
    )

    coverage = _harmonic_coverage(
        power,
        frequencies,
        f0,
    )

    rms_db = (
        20.0
        * np.log10(
            np.maximum(
                rms,
                1e-8,
            )
        )
    )

    reference_db = float(
        np.percentile(
            rms_db,
            95,
        )
    )
    active = (
        rms_db
        >= (
            reference_db
            - 42.0
        )
    )

    confidence_risk = np.clip(
        (
            0.62
            - voiced_probability
        )
        / 0.52,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    coverage_risk = np.clip(
        (
            0.88
            - coverage
        )
        / 0.48,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    missing_f0 = (
        ~np.isfinite(
            f0
        )
    ) & active

    jump_risk, jump_count = (
        _isolated_f0_jump_risk(
            f0
        )
    )

    risk = (
        confidence_risk
        * 0.62
        + coverage_risk
        * 0.38
    )

    risk = np.maximum(
        risk,
        jump_risk,
    )

    risk[
        missing_f0
    ] = np.maximum(
        risk[
            missing_f0
        ],
        0.95,
    )

    risk[
        ~active
    ] = 0.0

    # Extend short risky observations slightly so the bypass starts before
    # an unstable frame instead of reacting one frame too late.
    frame_seconds = (
        hop_length
        / float(
            sr
        )
    )
    prepost_radius = max(
        1,
        int(
            round(
                0.10
                / frame_seconds
            )
        ),
    )

    risk = _moving_max(
        risk,
        prepost_radius,
    )

    fallback = (
        _frame_risk_to_fallback(
            risk,
            sensitivity,
        )
    )

    smooth_radius = max(
        1,
        int(
            round(
                (
                    crossfade_ms
                    / 1000.0
                )
                / frame_seconds
                / 2.0
            )
        ),
    )

    fallback = _moving_average(
        fallback,
        smooth_radius,
    )
    fallback = np.clip(
        fallback,
        0.0,
        0.95,
    )

    times = (
        np.arange(
            count,
            dtype=np.float32,
        )
        * frame_seconds
    )

    duration = (
        float(
            y.size
        )
        / float(
            sr
        )
    )

    frame_duration = frame_seconds

    safe_seconds = float(
        np.sum(
            fallback
            < 0.15
        )
        * frame_duration
    )
    blend_seconds = float(
        np.sum(
            (
                fallback
                >= 0.15
            )
            & (
                fallback
                < 0.65
            )
        )
        * frame_duration
    )
    fallback_seconds = float(
        np.sum(
            fallback
            >= 0.65
        )
        * frame_duration
    )

    regions = _regions_from_mask(
        fallback,
        frame_seconds,
    )

    report = HarmonyGuardReport(
        duration_seconds=duration,
        sensitivity=sensitivity,
        crossfade_ms=crossfade_ms,
        safe_seconds=min(
            duration,
            safe_seconds,
        ),
        blend_seconds=min(
            duration,
            blend_seconds,
        ),
        fallback_seconds=min(
            duration,
            fallback_seconds,
        ),
        risky_region_count=len(
            regions
        ),
        f0_jump_count=jump_count,
        mean_risk=float(
            np.mean(
                risk
            )
        ),
        max_risk=float(
            np.max(
                risk
            )
        ),
        fallback_gain=1.0,
        alignment_ms=0.0,
        risky_regions=regions,
    )

    _emit(
        log_callback,
        (
            "[Harmony Guard] 분석 완료: "
            f"안전 {report.safe_seconds:.1f}s / "
            f"부분 Blend {report.blend_seconds:.1f}s / "
            f"Pitch-only 우회 {report.fallback_seconds:.1f}s / "
            f"위험 구간 {report.risky_region_count}개 / "
            f"F0 급변 {report.f0_jump_count}건"
        ),
    )

    for start, end in regions[
        :12
    ]:
        _emit(
            log_callback,
            (
                "[Harmony Guard] 위험 후보 "
                f"{start:.2f}s ~ {end:.2f}s"
            ),
        )

    if len(
        regions
    ) > 12:
        _emit(
            log_callback,
            (
                "[Harmony Guard] 위험 후보 "
                f"{len(regions) - 12}개 추가 생략"
            ),
        )

    return (
        times,
        fallback,
        report,
    )


def _resample_channels(
    data: np.ndarray,
    *,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    if original_sr == target_sr:
        return data.astype(
            np.float32,
            copy=False,
        )

    transposed = data.T

    try:
        resampled = librosa.resample(
            transposed,
            orig_sr=original_sr,
            target_sr=target_sr,
            axis=-1,
            res_type="soxr_hq",
        )
    except Exception:
        resampled = librosa.resample(
            transposed,
            orig_sr=original_sr,
            target_sr=target_sr,
            axis=-1,
            res_type="polyphase",
        )

    return np.asarray(
        resampled.T,
        dtype=np.float32,
    )


def _channel_match(
    data: np.ndarray,
    channels: int,
) -> np.ndarray:
    if data.shape[1] == channels:
        return data

    if data.shape[1] == 1:
        return np.repeat(
            data,
            channels,
            axis=1,
        )

    mono = np.mean(
        data,
        axis=1,
        keepdims=True,
    )

    return np.repeat(
        mono,
        channels,
        axis=1,
    )


def _rms_envelope(
    data: np.ndarray,
    sr: int,
    *,
    rate_hz: int = 100,
) -> np.ndarray:
    mono = np.mean(
        data,
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    hop = max(
        1,
        int(
            round(
                sr
                / rate_hz
            )
        ),
    )
    window = max(
        hop,
        int(
            round(
                sr
                * 0.04
            )
        ),
    )

    squared = (
        mono
        * mono
    )

    kernel = np.full(
        window,
        1.0
        / window,
        dtype=np.float32,
    )

    smooth = np.convolve(
        squared,
        kernel,
        mode="same",
    )

    envelope = np.sqrt(
        np.maximum(
            smooth[
                ::hop
            ],
            0.0,
        )
    )

    return envelope.astype(
        np.float32,
        copy=False,
    )


def _align_fallback_by_envelope(
    reference: np.ndarray,
    fallback: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float]:
    ref_env = _rms_envelope(
        reference,
        sr,
    )
    alt_env = _rms_envelope(
        fallback,
        sr,
    )

    count = min(
        ref_env.size,
        alt_env.size,
    )

    if count < 50:
        return (
            fallback,
            0.0,
        )

    ref_env = ref_env[
        :count
    ]
    alt_env = alt_env[
        :count
    ]

    ref_env = (
        ref_env
        - np.mean(
            ref_env
        )
    )
    alt_env = (
        alt_env
        - np.mean(
            alt_env
        )
    )

    ref_std = float(
        np.std(
            ref_env
        )
    )
    alt_std = float(
        np.std(
            alt_env
        )
    )

    if (
        ref_std <= 1e-8
        or alt_std <= 1e-8
    ):
        return (
            fallback,
            0.0,
        )

    ref_env /= ref_std
    alt_env /= alt_std

    max_lag = min(
        30,
        count
        // 10,
    )

    best_lag = 0
    best_score = -1.0

    for lag in range(
        -max_lag,
        max_lag + 1,
    ):
        if lag > 0:
            left = ref_env[
                :count - lag
            ]
            right = alt_env[
                lag:count
            ]
        elif lag < 0:
            shift = -lag
            left = ref_env[
                shift:count
            ]
            right = alt_env[
                :count - shift
            ]
        else:
            left = ref_env
            right = alt_env

        if left.size < 20:
            continue

        score = float(
            np.mean(
                left
                * right
            )
        )

        if score > best_score:
            best_score = score
            best_lag = lag

    if (
        best_lag == 0
        or best_score < 0.18
    ):
        return (
            fallback,
            0.0,
        )

    shift_samples = int(
        round(
            best_lag
            * sr
            / 100.0
        )
    )

    aligned = np.zeros_like(
        fallback
    )

    if shift_samples > 0:
        # Fallback is delayed; move it earlier.
        if shift_samples < fallback.shape[0]:
            aligned[
                :-shift_samples
            ] = fallback[
                shift_samples:
            ]
    else:
        shift = -shift_samples

        if shift < fallback.shape[0]:
            aligned[
                shift:
            ] = fallback[
                :-shift
            ]

    alignment_ms = (
        -shift_samples
        / float(
            sr
        )
        * 1000.0
    )

    return (
        aligned,
        alignment_ms,
    )



def _artifact_priority_combine(
    input_fallback: np.ndarray,
    artifact_fallback: np.ndarray,
    sensitivity: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    sensitivity = (
        sensitivity
        if sensitivity in SENSITIVITY_LABELS
        else "medium"
    )

    harmony_caps = {
        "low": 0.12,
        "medium": 0.22,
        "high": 0.32,
    }
    harmony_cap = float(harmony_caps[sensitivity])

    input_fallback = np.clip(
        np.asarray(input_fallback, dtype=np.float32),
        0.0,
        0.98,
    )
    artifact_fallback = np.clip(
        np.asarray(artifact_fallback, dtype=np.float32),
        0.0,
        0.98,
    )

    harmony_hint = np.clip(
        input_fallback / 0.92 * harmony_cap,
        0.0,
        harmony_cap,
    ).astype(np.float32, copy=False)

    input_norm = np.clip(
        input_fallback / 0.92,
        0.0,
        1.0,
    )
    artifact_norm = np.clip(
        artifact_fallback / 0.98,
        0.0,
        1.0,
    )
    agreement_boost = (
        0.14
        * input_norm
        * artifact_norm
    ).astype(np.float32, copy=False)

    artifact_confirmed = np.clip(
        artifact_fallback
        + agreement_boost,
        0.0,
        0.98,
    ).astype(np.float32, copy=False)

    combined = np.maximum(
        harmony_hint,
        artifact_confirmed,
    ).astype(np.float32, copy=False)

    return combined, harmony_hint, harmony_cap


def _manual_bypass_mask(
    frame_times: np.ndarray,
    ranges: list[tuple[float, float]] | None,
    *,
    crossfade_ms: int,
) -> np.ndarray:
    times = np.asarray(frame_times, dtype=np.float32)
    mask = np.zeros(times.size, dtype=np.float32)

    if times.size <= 0 or not ranges:
        return mask

    fade_seconds = max(
        0.02,
        min(
            2.0,
            float(crossfade_ms) / 1000.0,
        ),
    )

    for raw_start, raw_end in ranges:
        start = max(0.0, float(raw_start))
        end = max(start, float(raw_end))

        if end <= start:
            continue

        inside = (
            (times >= start)
            & (times <= end)
        )
        mask[inside] = 0.98

        pre = (
            (times >= start - fade_seconds)
            & (times < start)
        )
        if np.any(pre):
            x = (
                times[pre]
                - (start - fade_seconds)
            ) / fade_seconds
            mask[pre] = np.maximum(
                mask[pre],
                (
                    _smoothstep(x)
                    * 0.98
                ).astype(np.float32, copy=False),
            )

        post = (
            (times > end)
            & (times <= end + fade_seconds)
        )
        if np.any(post):
            x = 1.0 - (
                times[post]
                - end
            ) / fade_seconds
            mask[post] = np.maximum(
                mask[post],
                (
                    _smoothstep(x)
                    * 0.98
                ).astype(np.float32, copy=False),
            )

    return np.clip(
        mask,
        0.0,
        0.98,
    ).astype(np.float32, copy=False)


def blend_adaptive_vocals(
    original_vocal: str | Path,
    rvc_vocal: str | Path,
    pitch_only_vocal: str | Path,
    output_wav: str | Path,
    *,
    sensitivity: str = "medium",
    crossfade_ms: int = 500,
    auto_guard_enabled: bool = True,
    manual_bypass_ranges: list[tuple[float, float]] | None = None,
    log_callback: LogCallback | None = None,
) -> HarmonyGuardReport:
    original = Path(original_vocal).resolve()
    rvc_path = Path(rvc_vocal).resolve()
    pitch_path = Path(pitch_only_vocal).resolve()
    output = Path(output_wav).resolve()

    manual_ranges = [
        (
            max(0.0, float(start)),
            max(0.0, float(end)),
        )
        for start, end in (manual_bypass_ranges or [])
        if float(end) > float(start)
    ]

    rvc_data, rvc_sr = sf.read(
        str(rvc_path),
        dtype="float32",
        always_2d=True,
    )
    fallback_data, fallback_sr = sf.read(
        str(pitch_path),
        dtype="float32",
        always_2d=True,
    )

    if rvc_data.size <= 0:
        raise RuntimeError(
            "Adaptive Guard용 RVC 출력이 비어 있습니다."
        )
    if fallback_data.size <= 0:
        raise RuntimeError(
            "Adaptive Guard용 Pitch-only 출력이 비어 있습니다."
        )

    fallback_data = _resample_channels(
        fallback_data,
        original_sr=int(fallback_sr),
        target_sr=int(rvc_sr),
    )

    channels = max(
        rvc_data.shape[1],
        fallback_data.shape[1],
    )
    rvc_data = _channel_match(
        rvc_data,
        channels,
    ).astype(np.float32, copy=False)
    fallback_data = _channel_match(
        fallback_data,
        channels,
    ).astype(np.float32, copy=False)

    target_length = rvc_data.shape[0]

    if fallback_data.shape[0] < target_length:
        fallback_data = np.pad(
            fallback_data,
            (
                (
                    0,
                    target_length
                    - fallback_data.shape[0],
                ),
                (0, 0),
            ),
        )
    elif fallback_data.shape[0] > target_length:
        fallback_data = fallback_data[
            :target_length
        ]

    (
        fallback_data,
        alignment_ms,
    ) = _align_fallback_by_envelope(
        rvc_data,
        fallback_data,
        int(rvc_sr),
    )

    duration_seconds = (
        target_length
        / float(rvc_sr)
    )

    if auto_guard_enabled:
        (
            frame_times,
            input_fallback_ratio,
            report,
        ) = analyze_harmony_risk(
            original,
            sensitivity=sensitivity,
            crossfade_ms=crossfade_ms,
            log_callback=log_callback,
        )
    else:
        frame_seconds = 512.0 / 22050.0
        frame_count = max(
            2,
            int(
                np.ceil(
                    duration_seconds
                    / frame_seconds
                )
            )
            + 1,
        )
        frame_times = (
            np.arange(
                frame_count,
                dtype=np.float32,
            )
            * frame_seconds
        )
        input_fallback_ratio = np.zeros(
            frame_count,
            dtype=np.float32,
        )
        report = HarmonyGuardReport(
            duration_seconds=float(duration_seconds),
            sensitivity=(
                sensitivity
                if sensitivity in SENSITIVITY_LABELS
                else "medium"
            ),
            crossfade_ms=int(crossfade_ms),
            safe_seconds=float(duration_seconds),
            blend_seconds=0.0,
            fallback_seconds=0.0,
            risky_region_count=0,
            f0_jump_count=0,
            mean_risk=0.0,
            max_risk=0.0,
            fallback_gain=1.0,
            alignment_ms=0.0,
            risky_regions=[],
            artifact_guard_enabled=False,
        )
        _emit(
            log_callback,
            (
                "[Adaptive Guard v2.6] 자동 Guard OFF - "
                "수동 우회 구간만 적용합니다."
            ),
        )

    artifact_metrics = {
        "artifact_detected_seconds": 0.0,
        "artifact_strong_seconds": 0.0,
        "artifact_region_count": 0,
        "artifact_f0_mismatch_count": 0,
        "artifact_output_jump_count": 0,
        "artifact_voicing_mismatch_seconds": 0.0,
        "mean_artifact_risk": 0.0,
        "max_artifact_risk": 0.0,
    }

    artifact_on_input_grid = np.zeros(
        frame_times.size,
        dtype=np.float32,
    )

    if auto_guard_enabled:
        try:
            (
                artifact_times,
                artifact_fallback,
                _artifact_risk,
                artifact_metrics,
            ) = analyze_rvc_artifact_risk(
                rvc_data,
                fallback_data,
                int(rvc_sr),
                sensitivity=sensitivity,
                crossfade_ms=crossfade_ms,
                log_callback=log_callback,
            )

            if (
                frame_times.size > 1
                and artifact_times.size > 1
            ):
                artifact_on_input_grid = np.interp(
                    frame_times.astype(
                        np.float64,
                        copy=False,
                    ),
                    artifact_times.astype(
                        np.float64,
                        copy=False,
                    ),
                    artifact_fallback.astype(
                        np.float64,
                        copy=False,
                    ),
                    left=float(artifact_fallback[0]),
                    right=float(artifact_fallback[-1]),
                ).astype(np.float32, copy=False)

        except Exception as exc:
            _emit(
                log_callback,
                (
                    "[Artifact Guard] 2차 출력 검증 실패 - "
                    "Harmony 힌트 + 수동 우회만 사용합니다: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    if auto_guard_enabled:
        (
            auto_fallback,
            harmony_hint,
            harmony_cap,
        ) = _artifact_priority_combine(
            input_fallback_ratio,
            artifact_on_input_grid,
            sensitivity,
        )

        _emit(
            log_callback,
            (
                "[Adaptive Guard v2.6] Artifact 우선 모드: "
                f"Harmony-only 최대 우회 "
                f"{harmony_cap * 100.0:.0f}% / "
                "실제 Artifact 검출은 최대 98% 우회"
            ),
        )
    else:
        auto_fallback = np.zeros(
            frame_times.size,
            dtype=np.float32,
        )
        harmony_hint = np.zeros(
            frame_times.size,
            dtype=np.float32,
        )
        harmony_cap = 0.0

    manual_fallback = _manual_bypass_mask(
        frame_times,
        manual_ranges,
        crossfade_ms=int(crossfade_ms),
    )

    if manual_ranges:
        _emit(
            log_callback,
            (
                "[Manual Bypass] 수동 우회 "
                f"{len(manual_ranges)}개 구간 적용: "
                + ", ".join(
                    f"{start:.2f}-{end:.2f}s"
                    for start, end in manual_ranges[:12]
                )
                + (
                    f" 외 {len(manual_ranges) - 12}개"
                    if len(manual_ranges) > 12
                    else ""
                )
            ),
        )

    fallback_ratio = np.maximum(
        auto_fallback,
        manual_fallback,
    ).astype(np.float32, copy=False)

    fallback_ratio = np.clip(
        fallback_ratio,
        0.0,
        0.98,
    )

    rms_rvc = float(
        np.sqrt(
            np.mean(
                np.square(
                    rvc_data,
                    dtype=np.float64,
                )
            )
            + 1e-12
        )
    )
    rms_fallback = float(
        np.sqrt(
            np.mean(
                np.square(
                    fallback_data,
                    dtype=np.float64,
                )
            )
            + 1e-12
        )
    )

    gain = 1.0
    if rms_rvc > 1e-7 and rms_fallback > 1e-7:
        gain = max(
            0.55,
            min(
                1.80,
                rms_rvc / rms_fallback,
            ),
        )

    fallback_data = (
        fallback_data
        * float(gain)
    ).astype(np.float32, copy=False)

    if frame_times.size <= 1:
        sample_fallback = np.zeros(
            target_length,
            dtype=np.float32,
        )
    else:
        sample_times = (
            np.arange(
                target_length,
                dtype=np.float64,
            )
            / float(rvc_sr)
        )
        sample_fallback = np.interp(
            sample_times,
            frame_times.astype(
                np.float64,
                copy=False,
            ),
            fallback_ratio.astype(
                np.float64,
                copy=False,
            ),
            left=float(fallback_ratio[0]),
            right=float(fallback_ratio[-1]),
        ).astype(np.float32, copy=False)

    blend = sample_fallback[:, None]
    adaptive = (
        rvc_data
        * (1.0 - blend)
        + fallback_data
        * blend
    )

    peak = float(
        np.max(
            np.abs(adaptive)
        )
    )
    if peak > 0.98:
        adaptive = (
            adaptive
            * (0.98 / peak)
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sf.write(
        str(output),
        adaptive,
        int(rvc_sr),
        subtype="PCM_24",
    )

    report.guard_version = GUARD_VERSION
    report.fallback_gain = float(gain)
    report.alignment_ms = float(alignment_ms)
    report.artifact_guard_enabled = bool(
        auto_guard_enabled
    )
    report.artifact_priority_mode = True
    report.harmony_hint_cap = float(
        harmony_cap
    )
    report.artifact_detected_seconds = float(
        artifact_metrics[
            "artifact_detected_seconds"
        ]
    )
    report.artifact_strong_seconds = float(
        artifact_metrics[
            "artifact_strong_seconds"
        ]
    )
    report.artifact_region_count = int(
        artifact_metrics[
            "artifact_region_count"
        ]
    )
    report.artifact_f0_mismatch_count = int(
        artifact_metrics[
            "artifact_f0_mismatch_count"
        ]
    )
    report.artifact_output_jump_count = int(
        artifact_metrics[
            "artifact_output_jump_count"
        ]
    )
    report.artifact_voicing_mismatch_seconds = float(
        artifact_metrics[
            "artifact_voicing_mismatch_seconds"
        ]
    )
    report.mean_artifact_risk = float(
        artifact_metrics[
            "mean_artifact_risk"
        ]
    )
    report.max_artifact_risk = float(
        artifact_metrics[
            "max_artifact_risk"
        ]
    )
    report.manual_region_count = len(
        manual_ranges
    )
    report.manual_regions = list(
        manual_ranges
    )

    final_frame_seconds = (
        float(
            frame_times[1]
            - frame_times[0]
        )
        if frame_times.size > 1
        else 0.0
    )

    if final_frame_seconds > 0.0:
        report.safe_seconds = min(
            report.duration_seconds,
            float(
                np.sum(
                    fallback_ratio < 0.15
                )
                * final_frame_seconds
            ),
        )
        report.blend_seconds = min(
            report.duration_seconds,
            float(
                np.sum(
                    (
                        fallback_ratio >= 0.15
                    )
                    & (
                        fallback_ratio < 0.65
                    )
                )
                * final_frame_seconds
            ),
        )
        report.fallback_seconds = min(
            report.duration_seconds,
            float(
                np.sum(
                    fallback_ratio >= 0.65
                )
                * final_frame_seconds
            ),
        )
        report.manual_bypass_seconds = min(
            report.duration_seconds,
            float(
                np.sum(
                    manual_fallback >= 0.65
                )
                * final_frame_seconds
            ),
        )

        final_regions = _regions_from_mask(
            fallback_ratio,
            final_frame_seconds,
            threshold=0.35,
            min_seconds=0.08,
        )
        report.risky_regions = final_regions
        report.risky_region_count = len(
            final_regions
        )

    _emit(
        log_callback,
        (
            "[Adaptive Guard v2.6] 최종 적용: "
            f"강한 Pitch-only 우회 "
            f"{report.fallback_seconds:.1f}s / "
            f"부분 Blend {report.blend_seconds:.1f}s / "
            f"Artifact {report.artifact_detected_seconds:.1f}s / "
            f"수동 우회 {report.manual_bypass_seconds:.1f}s / "
            f"총 위험 구간 {report.risky_region_count}개"
        ),
    )
    _emit(
        log_callback,
        (
            "[Adaptive Guard v2.6] Adaptive vocal 생성 완료: "
            f"fallback gain={gain:.3f}x, "
            f"alignment={alignment_ms:+.1f}ms"
        ),
    )

    try:
        logs_dir = (
            Path(__file__).resolve().parent
            / "logs"
        )
        logs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        encoded = (
            json.dumps(
                asdict(report),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        for log_name in (
            "rvc_harmony_guard_last.json",
            "rvc_adaptive_guard_last.json",
        ):
            (
                logs_dir
                / log_name
            ).write_text(
                encoded,
                encoding="utf-8",
            )
    except OSError:
        pass

    return report

def self_test() -> dict[str, float]:
    sr = 22050
    seconds = 3.0
    time = (
        np.arange(
            int(
                sr
                * seconds
            )
        )
        / float(
            sr
        )
    )

    def harmonic_voice(
        fundamental: float,
    ) -> np.ndarray:
        signal = np.zeros_like(
            time,
            dtype=np.float64,
        )

        for harmonic in range(
            1,
            12,
        ):
            signal += (
                1.0
                / harmonic
            ) * np.sin(
                2.0
                * np.pi
                * fundamental
                * harmonic
                * time
            )

        peak = np.max(
            np.abs(
                signal
            )
        )

        return (
            signal
            / max(
                peak,
                1e-8,
            )
        ).astype(
            np.float32,
        )

    mono = harmonic_voice(
        220.0
    )
    harmony = (
        harmonic_voice(
            220.0
        )
        + 0.70
        * harmonic_voice(
            277.18
        )
        + 0.50
        * harmonic_voice(
            329.63
        )
    )
    harmony /= max(
        float(
            np.max(
                np.abs(
                    harmony
                )
            )
        ),
        1e-8,
    )

    # Same-key clean reference / RVC-good should yield almost no artifact
    # fallback.  RVC-bad contains a deliberate wrong-octave segment.
    reference = mono.copy()
    rvc_good = mono.copy()
    rvc_bad = mono.copy()

    bad_start = int(
        sr
        * 1.10
    )
    bad_end = int(
        sr
        * 1.55
    )

    wrong_note = harmonic_voice(
        329.63
    )
    rvc_bad[
        bad_start:bad_end
    ] = wrong_note[
        bad_start:bad_end
    ]

    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="adaptive_guard_selftest_"
    ) as temp_name:
        temp = Path(
            temp_name
        )
        mono_path = (
            temp
            / "mono.wav"
        )
        harmony_path = (
            temp
            / "harmony.wav"
        )

        sf.write(
            str(
                mono_path
            ),
            mono,
            sr,
        )
        sf.write(
            str(
                harmony_path
            ),
            harmony,
            sr,
        )

        _, mono_mask, _ = (
            analyze_harmony_risk(
                mono_path,
                sensitivity="medium",
                crossfade_ms=300,
            )
        )
        _, harmony_mask, _ = (
            analyze_harmony_risk(
                harmony_path,
                sensitivity="medium",
                crossfade_ms=300,
            )
        )

    _, clean_artifact_mask, _, clean_metrics = (
        analyze_rvc_artifact_risk(
            rvc_good[
                :,
                None,
            ],
            reference[
                :,
                None,
            ],
            sr,
            sensitivity="medium",
            crossfade_ms=300,
        )
    )
    _, bad_artifact_mask, _, bad_metrics = (
        analyze_rvc_artifact_risk(
            rvc_bad[
                :,
                None,
            ],
            reference[
                :,
                None,
            ],
            sr,
            sensitivity="medium",
            crossfade_ms=300,
        )
    )

    mono_value = float(
        np.mean(
            mono_mask
        )
    )
    harmony_value = float(
        np.mean(
            harmony_mask
        )
    )
    clean_artifact = float(
        np.mean(
            clean_artifact_mask
        )
    )
    bad_artifact = float(
        np.mean(
            bad_artifact_mask
        )
    )

    if not (
        harmony_value
        > mono_value
        + 0.15
    ):
        raise RuntimeError(
            "Harmony Guard self-test failed: "
            f"mono={mono_value:.3f}, harmony={harmony_value:.3f}"
        )

    if not (
        bad_artifact
        > clean_artifact
        + 0.12
        and bad_metrics[
            "artifact_detected_seconds"
        ]
        > clean_metrics[
            "artifact_detected_seconds"
        ]
        + 0.15
    ):
        raise RuntimeError(
            "Artifact Guard self-test failed: "
            f"clean={clean_artifact:.3f}, bad={bad_artifact:.3f}"
        )

    synthetic_harmony = np.full(
        100,
        0.92,
        dtype=np.float32,
    )
    synthetic_no_artifact = np.zeros(
        100,
        dtype=np.float32,
    )
    synthetic_artifact = np.full(
        100,
        0.90,
        dtype=np.float32,
    )

    harmony_only, _, medium_cap = (
        _artifact_priority_combine(
            synthetic_harmony,
            synthetic_no_artifact,
            "medium",
        )
    )
    artifact_confirmed, _, _ = (
        _artifact_priority_combine(
            synthetic_harmony,
            synthetic_artifact,
            "medium",
        )
    )

    if float(np.max(harmony_only)) > medium_cap + 1e-5:
        raise RuntimeError(
            "v2.6 Artifact-priority self-test failed: "
            "Harmony-only fallback exceeded cap."
        )

    if float(np.mean(artifact_confirmed)) <= 0.88:
        raise RuntimeError(
            "v2.6 Artifact-priority self-test failed: "
            "confirmed artifact was not strongly bypassed."
        )

    manual_times = np.arange(
        0.0,
        3.0,
        0.02,
        dtype=np.float32,
    )
    manual_mask = _manual_bypass_mask(
        manual_times,
        [(1.0, 1.5)],
        crossfade_ms=200,
    )

    if float(np.max(manual_mask)) < 0.97:
        raise RuntimeError(
            "v2.6 manual bypass self-test failed."
        )

    return {
        "mono_input_fallback": mono_value,
        "harmony_input_fallback": harmony_value,
        "clean_output_fallback": clean_artifact,
        "bad_output_fallback": bad_artifact,
        "bad_output_detected_seconds": float(
            bad_metrics[
                "artifact_detected_seconds"
            ]
        ),
        "v26_harmony_only_cap": float(
            np.max(harmony_only)
        ),
        "v26_confirmed_artifact": float(
            np.mean(artifact_confirmed)
        ),
        "v26_manual_peak": float(
            np.max(manual_mask)
        ),
    }

