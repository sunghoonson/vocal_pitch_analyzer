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

    resampled = librosa.resample(
        transposed,
        orig_sr=original_sr,
        target_sr=target_sr,
        axis=-1,
        res_type="kaiser_best",
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


def blend_adaptive_vocals(
    original_vocal: str | Path,
    rvc_vocal: str | Path,
    pitch_only_vocal: str | Path,
    output_wav: str | Path,
    *,
    sensitivity: str = "medium",
    crossfade_ms: int = 500,
    log_callback: LogCallback | None = None,
) -> HarmonyGuardReport:
    original = Path(
        original_vocal
    ).resolve()
    rvc_path = Path(
        rvc_vocal
    ).resolve()
    pitch_path = Path(
        pitch_only_vocal
    ).resolve()
    output = Path(
        output_wav
    ).resolve()

    (
        frame_times,
        fallback_ratio,
        report,
    ) = analyze_harmony_risk(
        original,
        sensitivity=sensitivity,
        crossfade_ms=crossfade_ms,
        log_callback=log_callback,
    )

    rvc_data, rvc_sr = sf.read(
        str(
            rvc_path
        ),
        dtype="float32",
        always_2d=True,
    )
    fallback_data, fallback_sr = sf.read(
        str(
            pitch_path
        ),
        dtype="float32",
        always_2d=True,
    )

    if rvc_data.size <= 0:
        raise RuntimeError(
            "Harmony Guard용 RVC 출력이 비어 있습니다."
        )

    if fallback_data.size <= 0:
        raise RuntimeError(
            "Harmony Guard용 Pitch-only 출력이 비어 있습니다."
        )

    fallback_data = _resample_channels(
        fallback_data,
        original_sr=int(
            fallback_sr
        ),
        target_sr=int(
            rvc_sr
        ),
    )

    channels = max(
        rvc_data.shape[1],
        fallback_data.shape[1],
    )

    rvc_data = _channel_match(
        rvc_data,
        channels,
    ).astype(
        np.float32,
        copy=False,
    )
    fallback_data = _channel_match(
        fallback_data,
        channels,
    ).astype(
        np.float32,
        copy=False,
    )

    target_length = (
        rvc_data.shape[0]
    )

    if fallback_data.shape[0] < target_length:
        fallback_data = np.pad(
            fallback_data,
            (
                (
                    0,
                    target_length
                    - fallback_data.shape[0],
                ),
                (
                    0,
                    0,
                ),
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
        int(
            rvc_sr
        ),
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

    if (
        rms_rvc > 1e-7
        and rms_fallback > 1e-7
    ):
        gain = max(
            0.55,
            min(
                1.80,
                rms_rvc
                / rms_fallback,
            ),
        )

    fallback_data = (
        fallback_data
        * float(
            gain
        )
    ).astype(
        np.float32,
        copy=False,
    )

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
            / float(
                rvc_sr
            )
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
            left=float(
                fallback_ratio[
                    0
                ]
            ),
            right=float(
                fallback_ratio[
                    -1
                ]
            ),
        ).astype(
            np.float32,
            copy=False,
        )

    blend = sample_fallback[
        :,
        None,
    ]

    adaptive = (
        rvc_data
        * (
            1.0
            - blend
        )
        + fallback_data
        * blend
    )

    peak = float(
        np.max(
            np.abs(
                adaptive
            )
        )
    )

    if peak > 0.98:
        adaptive = (
            adaptive
            * (
                0.98
                / peak
            )
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(
            output
        ),
        adaptive,
        int(
            rvc_sr
        ),
        subtype="PCM_24",
    )

    report.fallback_gain = float(
        gain
    )
    report.alignment_ms = float(
        alignment_ms
    )

    _emit(
        log_callback,
        (
            "[Harmony Guard] Adaptive vocal 생성 완료: "
            f"fallback gain={gain:.3f}x, "
            f"alignment={alignment_ms:+.1f}ms"
        ),
    )

    try:
        log_path = (
            Path(__file__).resolve().parent
            / "logs"
            / "rvc_harmony_guard_last.json"
        )
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        log_path.write_text(
            json.dumps(
                asdict(
                    report
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    return report


def self_test() -> dict[str, float]:
    sr = 22050
    seconds = 2.0
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

    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="harmony_guard_selftest_"
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

    if not (
        harmony_value
        > mono_value
        + 0.15
    ):
        raise RuntimeError(
            "Harmony Guard self-test failed: "
            f"mono={mono_value:.3f}, harmony={harmony_value:.3f}"
        )

    return {
        "mono_fallback": mono_value,
        "harmony_fallback": harmony_value,
    }
