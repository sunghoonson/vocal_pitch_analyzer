from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json
import math
import shutil

import librosa
import numpy as np
import soundfile as sf


# V27_LEAD_VOCAL_SELECTOR_PATCH

LogCallback = Callable[[str], None]


class LeadVocalSelectorError(RuntimeError):
    pass


@dataclass(slots=True)
class LeadSelectorReport:
    duration_seconds: float
    strength: str
    sample_rate: int
    channels: int
    active_seconds: float
    selected_seconds: float
    selected_ratio: float
    mean_lead_confidence: float
    mean_time_gate: float
    lead_rms_db: float
    residual_rms_db: float
    lead_energy_ratio: float
    debug_lead_path: str | None = None
    debug_nonlead_path: str | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def selector_cache_dir() -> Path:
    return (
        project_root()
        / "cache"
        / "rvc_lead_selector"
    )


def selector_log_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_lead_selector_last.log"
    )


def selector_json_path() -> Path:
    return (
        project_root()
        / "logs"
        / "rvc_lead_selector_last.json"
    )


def _emit(
    callback: LogCallback | None,
    message: str,
) -> None:
    text = str(
        message
    ).strip()

    if not text:
        return

    try:
        path = selector_log_path()
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                text
                + "\n"
            )
    except OSError:
        pass

    if callback is not None:
        callback(
            text
        )


def _reset_log() -> None:
    path = selector_log_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "",
            encoding="utf-8",
        )
    except OSError:
        pass


def _resample(
    data: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    if int(
        original_sr
    ) == int(
        target_sr
    ):
        return np.asarray(
            data,
            dtype=np.float32,
        )

    try:
        output = librosa.resample(
            np.asarray(
                data,
                dtype=np.float32,
            ),
            orig_sr=int(
                original_sr
            ),
            target_sr=int(
                target_sr
            ),
            res_type="soxr_hq",
        )
    except Exception:
        output = librosa.resample(
            np.asarray(
                data,
                dtype=np.float32,
            ),
            orig_sr=int(
                original_sr
            ),
            target_sr=int(
                target_sr
            ),
            res_type="polyphase",
        )

    return np.asarray(
        output,
        dtype=np.float32,
    )


def _moving_average(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if (
        radius <= 0
        or values.size <= 1
    ):
        return values.copy()

    width = (
        radius
        * 2
        + 1
    )
    kernel = (
        np.ones(
            width,
            dtype=np.float32,
        )
        / float(
            width
        )
    )

    padded = np.pad(
        values,
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
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if (
        radius <= 0
        or values.size <= 1
    ):
        return values.copy()

    padded = np.pad(
        values,
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    output = np.empty_like(
        values
    )

    for index in range(
        values.size
    ):
        output[
            index
        ] = np.max(
            padded[
                index:
                index
                + radius
                * 2
                + 1
            ]
        )

    return output


def _harmonic_coverage(
    power: np.ndarray,
    frequencies: np.ndarray,
    f0: np.ndarray,
) -> np.ndarray:
    frames = min(
        power.shape[
            1
        ],
        f0.size,
    )

    result = np.zeros(
        frames,
        dtype=np.float32,
    )

    if frames <= 0:
        return result

    total = np.sum(
        power[
            :,
            :frames,
        ],
        axis=0,
    ) + 1e-12

    bin_hz = float(
        frequencies[
            1
        ]
        - frequencies[
            0
        ]
    ) if frequencies.size > 1 else 1.0

    for frame in range(
        frames
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

        explained = 0.0
        max_harmonic = min(
            36,
            int(
                8000.0
                // fundamental
            ),
        )

        for harmonic in range(
            1,
            max_harmonic
            + 1,
        ):
            center = (
                fundamental
                * harmonic
            )

            if center > 8000.0:
                break

            width_hz = max(
                24.0,
                center
                * 0.018,
            )
            half_bins = max(
                1,
                int(
                    math.ceil(
                        width_hz
                        / max(
                            bin_hz,
                            1e-6,
                        )
                    )
                ),
            )
            center_bin = int(
                round(
                    center
                    / max(
                        bin_hz,
                        1e-6,
                    )
                )
            )

            lo = max(
                0,
                center_bin
                - half_bins,
            )
            hi = min(
                power.shape[
                    0
                ],
                center_bin
                + half_bins
                + 1,
            )

            if hi > lo:
                explained += float(
                    np.sum(
                        power[
                            lo:hi,
                            frame,
                        ]
                    )
                )

        result[
            frame
        ] = float(
            np.clip(
                explained
                / float(
                    total[
                        frame
                    ]
                ),
                0.0,
                1.0,
            )
        )

    return result


def _frame_center_score(
    audio: np.ndarray,
    sr: int,
    analysis_sr: int,
    *,
    frame_length: int,
    hop_length: int,
    target_frames: int,
) -> np.ndarray:
    if (
        audio.ndim != 2
        or audio.shape[
            1
        ] < 2
    ):
        return np.full(
            target_frames,
            0.72,
            dtype=np.float32,
        )

    left = audio[
        :,
        0
    ]
    right = audio[
        :,
        1
    ]

    mid = (
        left
        + right
    ) * 0.5
    side = (
        left
        - right
    ) * 0.5

    mid = _resample(
        mid,
        sr,
        analysis_sr,
    )
    side = _resample(
        side,
        sr,
        analysis_sr,
    )

    mid_rms = librosa.feature.rms(
        y=mid,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[
        0
    ]
    side_rms = librosa.feature.rms(
        y=side,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[
        0
    ]

    count = min(
        target_frames,
        mid_rms.size,
        side_rms.size,
    )

    result = np.full(
        target_frames,
        0.72,
        dtype=np.float32,
    )

    if count > 0:
        center = (
            mid_rms[
                :count
            ]
            / (
                mid_rms[
                    :count
                ]
                + side_rms[
                    :count
                ]
                + 1e-8
            )
        )
        result[
            :count
        ] = np.clip(
            center,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

    return result


def _strength_config(
    strength: str,
) -> dict[str, float]:
    key = (
        strength
        if strength
        in {
            "gentle",
            "balanced",
            "strict",
        }
        else "balanced"
    )

    return {
        "gentle": {
            "threshold": 0.46,
            "base_keep": 0.16,
            "unvoiced_keep": 0.62,
            "band_ratio": 0.024,
            "center_floor": 0.46,
        },
        "balanced": {
            "threshold": 0.56,
            "base_keep": 0.10,
            "unvoiced_keep": 0.50,
            "band_ratio": 0.019,
            "center_floor": 0.52,
        },
        "strict": {
            "threshold": 0.64,
            "base_keep": 0.055,
            "unvoiced_keep": 0.38,
            "band_ratio": 0.015,
            "center_floor": 0.58,
        },
    }[
        key
    ]


def _write_report(
    report: LeadSelectorReport,
) -> None:
    path = selector_json_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
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


def select_lead_vocal(
    input_vocal: str | Path,
    lead_output: str | Path,
    nonlead_output: str | Path,
    *,
    strength: str = "balanced",
    log_callback: LogCallback | None = None,
    save_debug_copy: bool = True,
) -> LeadSelectorReport:
    source = Path(
        input_vocal
    ).resolve()
    lead_path = Path(
        lead_output
    ).resolve()
    residual_path = Path(
        nonlead_output
    ).resolve()

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    config = _strength_config(
        strength
    )
    strength = (
        strength
        if strength
        in {
            "gentle",
            "balanced",
            "strict",
        }
        else "balanced"
    )

    _reset_log()
    _emit(
        log_callback,
        (
            "[Lead Selector] 메인 보컬 선별 시작 "
            f"(strength={strength})"
        ),
    )

    audio, sr = sf.read(
        str(
            source
        ),
        dtype="float32",
        always_2d=True,
    )

    if audio.size <= 0:
        raise LeadVocalSelectorError(
            "입력 보컬 stem이 비어 있습니다."
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    duration = (
        audio.shape[
            0
        ]
        / float(
            sr
        )
    )

    mono = np.mean(
        audio,
        axis=1,
        dtype=np.float32,
    )

    analysis_sr = 22050
    analysis = _resample(
        mono,
        int(
            sr
        ),
        analysis_sr,
    )

    frame_length = 2048
    hop_length = 512
    frame_seconds = (
        hop_length
        / float(
            analysis_sr
        )
    )

    f0, _, voiced_probability = (
        librosa.pyin(
            analysis,
            fmin=65.0,
            fmax=1400.0,
            sr=analysis_sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    )

    voiced_probability = np.nan_to_num(
        voiced_probability,
        nan=0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    magnitude = np.abs(
        librosa.stft(
            analysis,
            n_fft=frame_length,
            hop_length=hop_length,
            win_length=frame_length,
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

    rms = librosa.feature.rms(
        y=analysis,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[
        0
    ].astype(
        np.float32,
        copy=False,
    )

    flatness = librosa.feature.spectral_flatness(
        S=np.maximum(
            power,
            1e-12,
        ),
    )[
        0
    ].astype(
        np.float32,
        copy=False,
    )

    frequencies = librosa.fft_frequencies(
        sr=analysis_sr,
        n_fft=frame_length,
    )
    coverage = _harmonic_coverage(
        power,
        frequencies,
        f0,
    )

    count = min(
        f0.size,
        voiced_probability.size,
        rms.size,
        flatness.size,
        coverage.size,
    )

    f0 = f0[
        :count
    ]
    voiced_probability = voiced_probability[
        :count
    ]
    rms = rms[
        :count
    ]
    flatness = flatness[
        :count
    ]
    coverage = coverage[
        :count
    ]

    center_score = _frame_center_score(
        audio,
        int(
            sr
        ),
        analysis_sr,
        frame_length=frame_length,
        hop_length=hop_length,
        target_frames=count,
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
    ) if rms_db.size else -60.0

    active = (
        rms_db
        >= (
            reference_db
            - 42.0
        )
    )

    center_pref = np.clip(
        (
            center_score
            - float(
                config[
                    "center_floor"
                ]
            )
        )
        / 0.40,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    tonality = np.clip(
        (
            0.30
            - flatness
        )
        / 0.27,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    level_score = np.clip(
        (
            rms_db
            - (
                reference_db
                - 32.0
            )
        )
        / 24.0,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    lead_confidence = (
        voiced_probability
        * 0.36
        + coverage
        * 0.30
        + center_pref
        * 0.14
        + tonality
        * 0.12
        + level_score
        * 0.08
    ).astype(
        np.float32,
        copy=False,
    )

    valid_f0 = (
        np.isfinite(
            f0
        )
        & (
            f0
            > 0.0
        )
    )

    lead_confidence[
        ~active
    ] = 0.0

    lead_confidence[
        active
        & ~valid_f0
    ] *= 0.28

    threshold = float(
        config[
            "threshold"
        ]
    )

    confident = (
        active
        & valid_f0
        & (
            voiced_probability
            >= 0.34
        )
        & (
            coverage
            >= 0.32
        )
        & (
            lead_confidence
            >= threshold
        )
    )

    soft_gate = np.clip(
        (
            lead_confidence
            - (
                threshold
                - 0.13
            )
        )
        / 0.28,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    soft_gate[
        ~active
    ] = 0.0

    core = confident.astype(
        np.float32
    )

    near_radius = max(
        1,
        int(
            round(
                0.16
                / frame_seconds
            )
        ),
    )
    near = _moving_max(
        core,
        near_radius,
    )

    time_gate = np.maximum(
        soft_gate,
        (
            near
            * 0.74
            * active.astype(
                np.float32
            )
        ),
    )

    smooth_radius = max(
        1,
        int(
            round(
                0.055
                / frame_seconds
            )
        ),
    )
    time_gate = _moving_average(
        time_gate,
        smooth_radius,
    )
    time_gate = np.clip(
        time_gate,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # Spectral lead mask at source sample rate.
    # The selected dominant F0 gets harmonic bands; non-harmonic
    # energy remains mostly in the residual stem.
    # --------------------------------------------------------
    out_n_fft = (
        4096
        if int(
            sr
        ) >= 32000
        else 2048
    )
    out_hop = (
        out_n_fft
        // 4
    )
    out_bins = (
        out_n_fft
        // 2
        + 1
    )
    out_frame_count = (
        1
        + int(
            math.ceil(
                audio.shape[
                    0
                ]
                / float(
                    out_hop
                )
            )
        )
    )

    out_times = (
        np.arange(
            out_frame_count,
            dtype=np.float64,
        )
        * out_hop
        / float(
            sr
        )
    )
    analysis_times = (
        np.arange(
            count,
            dtype=np.float64,
        )
        * frame_seconds
    )

    if count > 1:
        gate_out = np.interp(
            out_times,
            analysis_times,
            time_gate.astype(
                np.float64,
                copy=False,
            ),
            left=float(
                time_gate[
                    0
                ]
            ),
            right=float(
                time_gate[
                    -1
                ]
            ),
        ).astype(
            np.float32,
            copy=False,
        )

        f0_fill = np.nan_to_num(
            f0,
            nan=0.0,
        ).astype(
            np.float64,
            copy=False,
        )
        f0_out = np.interp(
            out_times,
            analysis_times,
            f0_fill,
            left=0.0,
            right=0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        voiced_out = np.interp(
            out_times,
            analysis_times,
            voiced_probability.astype(
                np.float64,
                copy=False,
            ),
            left=0.0,
            right=0.0,
        ).astype(
            np.float32,
            copy=False,
        )
    else:
        gate_out = np.zeros(
            out_frame_count,
            dtype=np.float32,
        )
        f0_out = np.zeros(
            out_frame_count,
            dtype=np.float32,
        )
        voiced_out = np.zeros(
            out_frame_count,
            dtype=np.float32,
        )

    mask = np.zeros(
        (
            out_bins,
            out_frame_count,
        ),
        dtype=np.float32,
    )

    bin_hz = (
        float(
            sr
        )
        / float(
            out_n_fft
        )
    )
    base_keep = float(
        config[
            "base_keep"
        ]
    )
    unvoiced_keep = float(
        config[
            "unvoiced_keep"
        ]
    )
    band_ratio = float(
        config[
            "band_ratio"
        ]
    )

    for frame in range(
        out_frame_count
    ):
        gate = float(
            gate_out[
                frame
            ]
        )

        if gate <= 0.005:
            continue

        fundamental = float(
            f0_out[
                frame
            ]
        )

        if (
            fundamental <= 0.0
            or not math.isfinite(
                fundamental
            )
            or float(
                voiced_out[
                    frame
                ]
            ) < 0.12
        ):
            mask[
                :,
                frame
            ] = (
                gate
                * unvoiced_keep
            )
            continue

        harmonic_mask = np.zeros(
            out_bins,
            dtype=np.float32,
        )

        max_harmonic = min(
            32,
            int(
                9000.0
                // fundamental
            ),
        )

        for harmonic in range(
            1,
            max_harmonic
            + 1,
        ):
            center = (
                fundamental
                * harmonic
            )

            if center >= (
                float(
                    sr
                )
                * 0.48
            ):
                break

            width_hz = max(
                32.0,
                center
                * band_ratio,
            )
            center_bin = int(
                round(
                    center
                    / max(
                        bin_hz,
                        1e-6,
                    )
                )
            )
            half_bins = max(
                2,
                int(
                    math.ceil(
                        (
                            width_hz
                            * 2.6
                        )
                        / max(
                            bin_hz,
                            1e-6,
                        )
                    )
                ),
            )
            lo = max(
                0,
                center_bin
                - half_bins,
            )
            hi = min(
                out_bins,
                center_bin
                + half_bins
                + 1,
            )

            if hi <= lo:
                continue

            local_bins = np.arange(
                lo,
                hi,
                dtype=np.float32,
            )
            local_hz = (
                local_bins
                * bin_hz
            )
            weights = np.exp(
                -0.5
                * np.square(
                    (
                        local_hz
                        - center
                    )
                    / max(
                        width_hz,
                        1e-6,
                    )
                )
            ).astype(
                np.float32,
                copy=False,
            )

            harmonic_mask[
                lo:hi
            ] = np.maximum(
                harmonic_mask[
                    lo:hi
                ],
                weights,
            )

        spectral_keep = (
            base_keep
            + (
                1.0
                - base_keep
            )
            * harmonic_mask
        )

        mask[
            :,
            frame
        ] = (
            gate
            * spectral_keep
        )

    if mask.shape[
        1
    ] > 2:
        original_mask = mask.copy()
        mask[
            :,
            1:-1
        ] = (
            original_mask[
                :,
                :-2
            ]
            + original_mask[
                :,
                2:
            ]
            + original_mask[
                :,
                1:-1
            ]
            * 2.0
        ) * 0.25

    lead_channels: list[
        np.ndarray
    ] = []

    for channel in range(
        audio.shape[
            1
        ]
    ):
        spectrum = librosa.stft(
            audio[
                :,
                channel
            ],
            n_fft=out_n_fft,
            hop_length=out_hop,
            win_length=out_n_fft,
            center=True,
        )

        usable_frames = min(
            spectrum.shape[
                1
            ],
            mask.shape[
                1
            ],
        )

        selected = spectrum.copy()
        selected[
            :,
            :usable_frames
        ] *= mask[
            :,
            :usable_frames
        ]

        if spectrum.shape[
            1
        ] > usable_frames:
            selected[
                :,
                usable_frames:
            ] = 0.0

        lead_channel = librosa.istft(
            selected,
            hop_length=out_hop,
            win_length=out_n_fft,
            center=True,
            length=audio.shape[
                0
            ],
        ).astype(
            np.float32,
            copy=False,
        )

        lead_channels.append(
            lead_channel
        )

    lead = np.stack(
        lead_channels,
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    residual = (
        audio
        - lead
    ).astype(
        np.float32,
        copy=False,
    )

    lead_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    residual_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(
            lead_path
        ),
        lead,
        int(
            sr
        ),
        subtype="PCM_24",
    )
    sf.write(
        str(
            residual_path
        ),
        residual,
        int(
            sr
        ),
        subtype="PCM_24",
    )

    lead_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    lead,
                    dtype=np.float64,
                )
            )
            + 1e-12
        )
    )
    residual_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    residual,
                    dtype=np.float64,
                )
            )
            + 1e-12
        )
    )
    source_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    audio,
                    dtype=np.float64,
                )
            )
            + 1e-12
        )
    )

    active_seconds = float(
        np.sum(
            active
        )
        * frame_seconds
    )
    selected_seconds = min(
        float(
            duration
        ),
        float(
            np.sum(
                time_gate
                >= 0.35
            )
            * frame_seconds
        ),
    )

    debug_lead = None
    debug_nonlead = None

    if save_debug_copy:
        cache = selector_cache_dir()
        cache.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_lead = (
            cache
            / "last_lead_candidate.wav"
        )
        debug_nonlead = (
            cache
            / "last_nonlead_residual.wav"
        )

        try:
            shutil.copy2(
                lead_path,
                debug_lead,
            )
            shutil.copy2(
                residual_path,
                debug_nonlead,
            )
        except OSError:
            debug_lead = None
            debug_nonlead = None

    report = LeadSelectorReport(
        duration_seconds=float(
            duration
        ),
        strength=strength,
        sample_rate=int(
            sr
        ),
        channels=int(
            audio.shape[
                1
            ]
        ),
        active_seconds=active_seconds,
        selected_seconds=selected_seconds,
        selected_ratio=(
            selected_seconds
            / max(
                duration,
                1e-6,
            )
        ),
        mean_lead_confidence=float(
            np.mean(
                lead_confidence[
                    active
                ]
            )
        ) if np.any(
            active
        ) else 0.0,
        mean_time_gate=float(
            np.mean(
                time_gate
            )
        ) if time_gate.size else 0.0,
        lead_rms_db=float(
            20.0
            * math.log10(
                max(
                    lead_rms,
                    1e-8,
                )
            )
        ),
        residual_rms_db=float(
            20.0
            * math.log10(
                max(
                    residual_rms,
                    1e-8,
                )
            )
        ),
        lead_energy_ratio=float(
            np.clip(
                (
                    lead_rms
                    * lead_rms
                )
                / max(
                    source_rms
                    * source_rms,
                    1e-12,
                ),
                0.0,
                1.5,
            )
        ),
        debug_lead_path=(
            str(
                debug_lead
            )
            if debug_lead
            is not None
            else None
        ),
        debug_nonlead_path=(
            str(
                debug_nonlead
            )
            if debug_nonlead
            is not None
            else None
        ),
    )

    _write_report(
        report
    )

    _emit(
        log_callback,
        (
            "[Lead Selector] 완료: "
            f"활성 보컬 {report.active_seconds:.1f}s / "
            f"Lead 선택 {report.selected_seconds:.1f}s "
            f"({report.selected_ratio * 100.0:.1f}%) / "
            f"Lead energy {report.lead_energy_ratio * 100.0:.1f}%"
        ),
    )

    if report.debug_lead_path:
        _emit(
            log_callback,
            (
                "[Lead Selector] 마지막 Lead stem: "
                + report.debug_lead_path
            ),
        )
    if report.debug_nonlead_path:
        _emit(
            log_callback,
            (
                "[Lead Selector] 마지막 Non-lead stem: "
                + report.debug_nonlead_path
            ),
        )

    return report


def self_test() -> dict[str, float]:
    sr = 22050
    duration = 2.5
    time = (
        np.arange(
            int(
                sr
                * duration
            ),
            dtype=np.float64,
        )
        / float(
            sr
        )
    )

    def harmonic_voice(
        fundamental: float,
        phase: float = 0.0,
    ) -> np.ndarray:
        result = np.zeros_like(
            time
        )

        for harmonic in range(
            1,
            10,
        ):
            result += (
                1.0
                / harmonic
            ) * np.sin(
                2.0
                * np.pi
                * fundamental
                * harmonic
                * time
                + phase
            )

        result /= max(
            float(
                np.max(
                    np.abs(
                        result
                    )
                )
            ),
            1e-8,
        )

        return result.astype(
            np.float32
        )

    lead = (
        harmonic_voice(
            220.0
        )
        * 0.62
    )
    backing = (
        harmonic_voice(
            277.18,
            phase=0.4,
        )
        * 0.35
    )

    # Lead is centered, backing is deliberately wide/opposite-phase.
    stereo = np.stack(
        [
            lead
            + backing,
            lead
            - backing,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    import tempfile

    with tempfile.TemporaryDirectory(
        prefix="lead_selector_selftest_"
    ) as temp_name:
        temp = Path(
            temp_name
        )
        source = (
            temp
            / "source.wav"
        )
        lead_out = (
            temp
            / "lead.wav"
        )
        residual_out = (
            temp
            / "residual.wav"
        )

        sf.write(
            str(
                source
            ),
            stereo,
            sr,
        )

        report = select_lead_vocal(
            source,
            lead_out,
            residual_out,
            strength="balanced",
            save_debug_copy=False,
        )

        selected, _ = sf.read(
            str(
                lead_out
            ),
            dtype="float32",
            always_2d=True,
        )
        residual, _ = sf.read(
            str(
                residual_out
            ),
            dtype="float32",
            always_2d=True,
        )

    selected_mono = np.mean(
        selected,
        axis=1,
    )
    residual_side = (
        residual[
            :,
            0
        ]
        - residual[
            :,
            1
        ]
    ) * 0.5

    def corr(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        count = min(
            a.size,
            b.size,
        )
        a = a[
            :count
        ].astype(
            np.float64,
            copy=False,
        )
        b = b[
            :count
        ].astype(
            np.float64,
            copy=False,
        )

        a -= np.mean(
            a
        )
        b -= np.mean(
            b
        )

        denom = (
            np.linalg.norm(
                a
            )
            * np.linalg.norm(
                b
            )
            + 1e-12
        )

        return float(
            np.dot(
                a,
                b,
            )
            / denom
        )

    lead_corr = corr(
        selected_mono,
        lead,
    )
    backing_corr = corr(
        residual_side,
        backing,
    )

    if lead_corr < 0.72:
        raise RuntimeError(
            "Lead Selector self-test failed: "
            f"lead_corr={lead_corr:.3f}"
        )
    if backing_corr < 0.72:
        raise RuntimeError(
            "Lead Selector self-test failed: "
            f"backing_corr={backing_corr:.3f}"
        )

    return {
        "lead_correlation": lead_corr,
        "backing_residual_correlation": backing_corr,
        "selected_ratio": report.selected_ratio,
        "lead_energy_ratio": report.lead_energy_ratio,
    }
