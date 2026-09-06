from __future__ import annotations

# V32_RVC_F0_STABILITY_GUARD_PATCH
#
# Project-local RVC inference wrapper.
#
# The pinned upstream RVC CLI does not expose an external F0-file option.
# Instead of editing tools/rvc in-place, this wrapper monkey-patches
# Pipeline.get_f0() before infer.cli creates VC/Pipeline instances.
#
# Normal RMVPE frames are preserved.
# Only strong downward subharmonic/octave-collapse candidates are fused
# toward a high-confidence pYIN reference track.

from pathlib import Path
import csv
import json
import math
import os
import sys
import traceback

import librosa
import numpy as np


STABILITY_MODES: dict[str, dict[str, float]] = {
    "conservative": {
        "pyin_probability": 0.68,
        "mismatch_semitones": 7.0,
        "ratio_max": 0.50,
        "active_margin_db": 40.0,
    },
    "balanced": {
        "pyin_probability": 0.52,
        "mismatch_semitones": 5.0,
        "ratio_max": 0.58,
        "active_margin_db": 44.0,
    },
    "strong": {
        "pyin_probability": 0.40,
        "mismatch_semitones": 3.5,
        "ratio_max": 0.66,
        "active_margin_db": 48.0,
    },
}


def _mode_name() -> str:
    value = str(
        os.environ.get(
            "VPA_F0_STABILITY_MODE",
            "balanced",
        )
        or "balanced"
    ).strip().lower()

    if value not in STABILITY_MODES:
        return "balanced"

    return value


def _enabled() -> bool:
    return str(
        os.environ.get(
            "VPA_F0_STABILITY",
            "1",
        )
    ).strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }


def _align_track(
    values: np.ndarray,
    target_count: int,
) -> np.ndarray:
    source = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    target_count = max(
        0,
        int(
            target_count
        ),
    )

    if target_count <= 0:
        return np.zeros(
            0,
            dtype=np.float32,
        )

    if source.size == target_count:
        return source.astype(
            np.float32,
            copy=True,
        )

    if source.size <= 0:
        return np.full(
            target_count,
            np.nan,
            dtype=np.float32,
        )

    if source.size == 1:
        return np.full(
            target_count,
            float(
                source[
                    0
                ]
            ),
            dtype=np.float32,
        )

    source_x = np.linspace(
        0.0,
        1.0,
        source.size,
        dtype=np.float64,
    )
    target_x = np.linspace(
        0.0,
        1.0,
        target_count,
        dtype=np.float64,
    )

    finite = np.isfinite(
        source
    )

    if not np.any(
        finite
    ):
        return np.full(
            target_count,
            np.nan,
            dtype=np.float32,
        )

    result = np.interp(
        target_x,
        source_x[
            finite
        ],
        source[
            finite
        ],
    )

    # Preserve invalid edges rather than extrapolating voiced pitch
    # far into silence.
    first_valid = int(
        np.argmax(
            finite
        )
    )
    last_valid = int(
        source.size
        - 1
        - np.argmax(
            finite[::-1]
        )
    )

    first_t = (
        first_valid
        / max(
            1,
            source.size - 1,
        )
    )
    last_t = (
        last_valid
        / max(
            1,
            source.size - 1,
        )
    )

    result[
        target_x < first_t
    ] = np.nan
    result[
        target_x > last_t
    ] = np.nan

    return result.astype(
        np.float32,
        copy=False,
    )


def _moving_average(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if (
        radius <= 0
        or array.size <= 2
    ):
        return array.astype(
            np.float32,
            copy=True,
        )

    radius = int(
        radius
    )
    kernel_size = (
        radius * 2
        + 1
    )
    kernel = np.full(
        kernel_size,
        1.0 / kernel_size,
        dtype=np.float32,
    )

    padded = np.pad(
        array,
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
    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if (
        radius <= 0
        or array.size <= 2
    ):
        return array.astype(
            np.float32,
            copy=True,
        )

    radius = int(
        radius
    )
    padded = np.pad(
        array,
        (
            radius,
            radius,
        ),
        mode="edge",
    )

    views = [
        padded[
            offset:
            offset
            + array.size
        ]
        for offset in range(
            radius * 2
            + 1
        )
    ]

    return np.maximum.reduce(
        views
    ).astype(
        np.float32,
        copy=False,
    )


def _midi(
    f0: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        f0,
        dtype=np.float32,
    )

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(
            values
        )
        & (
            values
            > 0.0
        )
    )

    result[
        valid
    ] = (
        69.0
        + 12.0
        * np.log2(
            values[
                valid
            ]
            / 440.0
        )
    )

    return result


def _coarse_from_f0(
    f0: np.ndarray,
) -> np.ndarray:
    f0 = np.asarray(
        f0,
        dtype=np.float32,
    )

    f0_min = 50.0
    f0_max = 1100.0

    f0_mel_min = (
        1127.0
        * np.log(
            1.0
            + f0_min
            / 700.0
        )
    )
    f0_mel_max = (
        1127.0
        * np.log(
            1.0
            + f0_max
            / 700.0
        )
    )

    safe = np.nan_to_num(
        f0,
        nan=0.0,
        posinf=f0_max,
        neginf=0.0,
    )

    f0_mel = (
        1127.0
        * np.log(
            1.0
            + np.maximum(
                safe,
                0.0,
            )
            / 700.0
        )
    )

    positive = (
        f0_mel
        > 0.0
    )

    f0_mel[
        positive
    ] = (
        (
            f0_mel[
                positive
            ]
            - f0_mel_min
        )
        * 254.0
        / (
            f0_mel_max
            - f0_mel_min
        )
        + 1.0
    )

    f0_mel[
        f0_mel <= 1.0
    ] = 1.0
    f0_mel[
        f0_mel > 255.0
    ] = 255.0

    return np.rint(
        f0_mel
    ).astype(
        np.int32
    )


def _regions(
    weight: np.ndarray,
    frame_seconds: float,
    *,
    threshold: float = 0.45,
) -> list[tuple[float, float]]:
    mask = (
        np.asarray(
            weight
        )
        >= float(
            threshold
        )
    )

    result: list[
        tuple[
            float,
            float,
        ]
    ] = []

    start: int | None = None

    for index, active in enumerate(
        mask
    ):
        if (
            active
            and start is None
        ):
            start = index

        elif (
            not active
            and start is not None
        ):
            result.append(
                (
                    start
                    * frame_seconds,
                    index
                    * frame_seconds,
                )
            )
            start = None

    if start is not None:
        result.append(
            (
                start
                * frame_seconds,
                mask.size
                * frame_seconds,
            )
        )

    return result


def stabilize_f0_track(
    rmvpe_f0: np.ndarray,
    reference_f0: np.ndarray,
    reference_probability: np.ndarray,
    active_mask: np.ndarray,
    *,
    mode: str = "balanced",
    frame_seconds: float = 0.01,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Correct only obvious downward RMVPE subharmonic collapse.

    A reference disagreement by itself is NOT enough.
    RMVPE must also be substantially below the pYIN reference by a
    frequency-ratio gate, which avoids replacing a valid lead with a
    stronger upper harmony tracked by pYIN.
    """
    mode = (
        mode
        if mode in STABILITY_MODES
        else "balanced"
    )
    cfg = STABILITY_MODES[
        mode
    ]

    rmvpe = np.asarray(
        rmvpe_f0,
        dtype=np.float32,
    ).reshape(-1)
    reference = np.asarray(
        reference_f0,
        dtype=np.float32,
    ).reshape(-1)
    probability = np.asarray(
        reference_probability,
        dtype=np.float32,
    ).reshape(-1)
    active = np.asarray(
        active_mask,
        dtype=bool,
    ).reshape(-1)

    count = min(
        rmvpe.size,
        reference.size,
        probability.size,
        active.size,
    )

    rmvpe = rmvpe[
        :count
    ]
    reference = reference[
        :count
    ]
    probability = probability[
        :count
    ]
    active = active[
        :count
    ]

    corrected = rmvpe.astype(
        np.float32,
        copy=True,
    )

    rmvpe_midi = _midi(
        rmvpe
    )
    reference_midi = _midi(
        reference
    )

    valid = (
        np.isfinite(
            rmvpe_midi
        )
        & np.isfinite(
            reference_midi
        )
        & (
            rmvpe
            > 0.0
        )
        & (
            reference
            > 0.0
        )
    )

    downward_mismatch = np.zeros(
        count,
        dtype=np.float32,
    )

    downward_mismatch[
        valid
    ] = (
        reference_midi[
            valid
        ]
        - rmvpe_midi[
            valid
        ]
    )

    ratio = np.ones(
        count,
        dtype=np.float32,
    )

    ratio[
        valid
    ] = (
        rmvpe[
            valid
        ]
        / np.maximum(
            reference[
                valid
            ],
            1e-6,
        )
    )

    candidate = (
        valid
        & active
        & (
            probability
            >= float(
                cfg[
                    "pyin_probability"
                ]
            )
        )
        & (
            downward_mismatch
            >= float(
                cfg[
                    "mismatch_semitones"
                ]
            )
        )
        & (
            ratio
            <= float(
                cfg[
                    "ratio_max"
                ]
            )
        )
    )

    # Strong absolute-low guard:
    # when RMVPE falls into sub-75 Hz territory while a confident
    # reference is clearly singing above 120 Hz, it is almost certainly
    # the collapse pattern observed in the user's failed RVC sections.
    absolute_low = (
        valid
        & active
        & (
            probability
            >= max(
                0.44,
                float(
                    cfg[
                        "pyin_probability"
                    ]
                )
                - 0.08,
            )
        )
        & (
            rmvpe
            < 75.0
        )
        & (
            reference
            > 120.0
        )
        & (
            downward_mismatch
            >= 5.0
        )
    )

    candidate = (
        candidate
        | absolute_low
    )

    raw_weight = candidate.astype(
        np.float32
    )

    # Include 20 ms around a detected collapse and crossfade in log-F0
    # over roughly 50-70 ms rather than making a hard pitch boundary.
    expanded = _moving_max(
        raw_weight,
        2,
    )

    weight = _moving_average(
        expanded,
        3,
    )

    weight = np.clip(
        weight,
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    blend_valid = (
        valid
        & (
            weight
            > 0.0
        )
    )

    if np.any(
        blend_valid
    ):
        rm_midi = rmvpe_midi[
            blend_valid
        ]
        ref_midi = reference_midi[
            blend_valid
        ]
        w = weight[
            blend_valid
        ]

        blended_midi = (
            rm_midi
            * (
                1.0
                - w
            )
            + ref_midi
            * w
        )

        corrected[
            blend_valid
        ] = (
            440.0
            * np.power(
                2.0,
                (
                    blended_midi
                    - 69.0
                )
                / 12.0,
            )
        ).astype(
            np.float32,
            copy=False,
        )

    regions = _regions(
        weight,
        frame_seconds,
    )

    strong_frames = int(
        np.sum(
            weight
            >= 0.45
        )
    )

    candidate_frames = int(
        np.sum(
            candidate
        )
    )

    max_downward = float(
        np.max(
            downward_mismatch[
                candidate
            ]
        )
    ) if np.any(
        candidate
    ) else 0.0

    summary = {
        "mode": mode,
        "frame_seconds": float(
            frame_seconds
        ),
        "candidate_frames": candidate_frames,
        "corrected_frames": strong_frames,
        "corrected_seconds": float(
            strong_frames
            * frame_seconds
        ),
        "region_count": len(
            regions
        ),
        "regions": [
            [
                float(
                    start
                ),
                float(
                    end
                ),
            ]
            for start, end in regions
        ],
        "max_downward_mismatch_semitones": max_downward,
        "thresholds": dict(
            cfg
        ),
    }

    return (
        corrected,
        weight,
        summary,
    )


def _reference_pyin(
    x: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    target_count: int,
    active_margin_db: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    audio = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(-1)

    frame_length = 2048

    f0, _voiced, probability = (
        librosa.pyin(
            audio,
            fmin=55.0,
            fmax=1100.0,
            sr=int(
                sr
            ),
            frame_length=frame_length,
            hop_length=int(
                hop_length
            ),
            center=True,
        )
    )

    probability = np.nan_to_num(
        probability,
        nan=0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=int(
            hop_length
        ),
        center=True,
    )[
        0
    ]

    f0 = _align_track(
        np.asarray(
            f0,
            dtype=np.float32,
        ),
        target_count,
    )
    probability = _align_track(
        probability,
        target_count,
    )
    rms = _align_track(
        rms,
        target_count,
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

    finite_db = rms_db[
        np.isfinite(
            rms_db
        )
    ]

    reference_db = float(
        np.percentile(
            finite_db,
            95,
        )
    ) if finite_db.size else -60.0

    active = (
        rms_db
        >= (
            reference_db
            - float(
                active_margin_db
            )
        )
    )

    return (
        f0,
        probability,
        active,
    )


def _write_diagnostics(
    *,
    rmvpe_f0: np.ndarray,
    reference_f0: np.ndarray,
    reference_probability: np.ndarray,
    corrected_f0: np.ndarray,
    target_f0: np.ndarray,
    weight: np.ndarray,
    summary: dict,
    frame_seconds: float,
    pad_seconds: float,
    unpadded_duration: float,
) -> None:
    csv_value = str(
        os.environ.get(
            "VPA_F0_DIAG_CSV",
            "",
        )
        or ""
    ).strip()
    json_value = str(
        os.environ.get(
            "VPA_F0_DIAG_JSON",
            "",
        )
        or ""
    ).strip()

    count = min(
        len(
            rmvpe_f0
        ),
        len(
            reference_f0
        ),
        len(
            reference_probability
        ),
        len(
            corrected_f0
        ),
        len(
            target_f0
        ),
        len(
            weight
        ),
    )

    times = (
        np.arange(
            count,
            dtype=np.float64,
        )
        * float(
            frame_seconds
        )
        - float(
            pad_seconds
        )
    )

    visible = (
        times
        >= 0.0
    )

    if unpadded_duration > 0.0:
        visible &= (
            times
            <= unpadded_duration
            + 0.02
        )

    rmvpe = np.asarray(
        rmvpe_f0
    )[
        :count
    ]
    reference = np.asarray(
        reference_f0
    )[
        :count
    ]
    probability = np.asarray(
        reference_probability
    )[
        :count
    ]
    corrected = np.asarray(
        corrected_f0
    )[
        :count
    ]
    target = np.asarray(
        target_f0
    )[
        :count
    ]
    weight_values = np.asarray(
        weight
    )[
        :count
    ]

    if csv_value:
        path = Path(
            csv_value
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.writer(
                handle
            )
            writer.writerow(
                [
                    "time_seconds",
                    "rmvpe_source_hz",
                    "reference_pyin_hz",
                    "reference_probability",
                    "downward_mismatch_semitones",
                    "corrected_source_hz",
                    "target_f0_hz",
                    "correction_weight",
                    "corrected",
                ]
            )

            for index in np.flatnonzero(
                visible
            ):
                rm = float(
                    rmvpe[
                        index
                    ]
                )
                ref = float(
                    reference[
                        index
                    ]
                )

                mismatch = ""

                if (
                    math.isfinite(
                        rm
                    )
                    and rm > 0.0
                    and math.isfinite(
                        ref
                    )
                    and ref > 0.0
                ):
                    mismatch = (
                        12.0
                        * math.log2(
                            ref
                            / rm
                        )
                    )

                writer.writerow(
                    [
                        f"{times[index]:.4f}",
                        (
                            f"{rm:.4f}"
                            if math.isfinite(
                                rm
                            )
                            else ""
                        ),
                        (
                            f"{ref:.4f}"
                            if math.isfinite(
                                ref
                            )
                            else ""
                        ),
                        f"{float(probability[index]):.4f}",
                        (
                            f"{float(mismatch):.4f}"
                            if mismatch != ""
                            else ""
                        ),
                        f"{float(corrected[index]):.4f}",
                        f"{float(target[index]):.4f}",
                        f"{float(weight_values[index]):.4f}",
                        (
                            "1"
                            if float(
                                weight_values[
                                    index
                                ]
                            )
                            >= 0.45
                            else "0"
                        ),
                    ]
                )

    if json_value:
        path = Path(
            json_value
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = dict(
            summary
        )
        data.update(
            {
                "stability_guard_version": "3.2",
                "enabled": True,
                "frame_count": int(
                    count
                ),
                "pad_seconds": float(
                    pad_seconds
                ),
                "unpadded_duration": float(
                    unpadded_duration
                ),
                "csv_path": (
                    str(
                        Path(
                            csv_value
                        )
                    )
                    if csv_value
                    else None
                ),
            }
        )

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def install_f0_stability_patch() -> None:
    from infer.vc.pipeline import Pipeline

    original_get_f0 = (
        Pipeline.get_f0
    )

    if getattr(
        original_get_f0,
        "_vpa_f0_stability",
        False,
    ):
        return

    def stable_get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
    ):
        if (
            not _enabled()
            or str(
                f0_method
            ).lower()
            != "rmvpe"
        ):
            return original_get_f0(
                self,
                x,
                p_len,
                f0_up_key,
                f0_method,
            )

        # Ask upstream RMVPE for UNTRANSPOSED F0 first.
        _unused_coarse, rmvpe_source = (
            original_get_f0(
                self,
                x,
                p_len,
                0,
                f0_method,
            )
        )

        rmvpe_source = np.asarray(
            rmvpe_source,
            dtype=np.float32,
        )[
            :int(
                p_len
            )
        ]

        mode = _mode_name()
        cfg = STABILITY_MODES[
            mode
        ]

        try:
            reference_f0, probability, active = (
                _reference_pyin(
                    np.asarray(
                        x,
                        dtype=np.float32,
                    ),
                    sr=int(
                        self.sr
                    ),
                    hop_length=int(
                        self.window
                    ),
                    target_count=rmvpe_source.size,
                    active_margin_db=float(
                        cfg[
                            "active_margin_db"
                        ]
                    ),
                )
            )

            corrected_source, weight, summary = (
                stabilize_f0_track(
                    rmvpe_source,
                    reference_f0,
                    probability,
                    active,
                    mode=mode,
                    frame_seconds=(
                        float(
                            self.window
                        )
                        / float(
                            self.sr
                        )
                    ),
                )
            )

        except Exception:
            print(
                "[F0 Stability Guard] reference analysis failed; "
                "falling back to raw RMVPE.",
                file=sys.stderr,
            )
            traceback.print_exc()

            corrected_source = (
                rmvpe_source
            )
            weight = np.zeros(
                rmvpe_source.size,
                dtype=np.float32,
            )
            reference_f0 = np.full(
                rmvpe_source.size,
                np.nan,
                dtype=np.float32,
            )
            probability = np.zeros(
                rmvpe_source.size,
                dtype=np.float32,
            )
            summary = {
                "mode": mode,
                "candidate_frames": 0,
                "corrected_frames": 0,
                "corrected_seconds": 0.0,
                "region_count": 0,
                "regions": [],
                "max_downward_mismatch_semitones": 0.0,
                "fallback_raw_rmvpe": True,
            }

        shift_ratio = (
            2.0
            ** (
                float(
                    f0_up_key
                )
                / 12.0
            )
        )

        target_f0 = (
            corrected_source
            * shift_ratio
        ).astype(
            np.float32,
            copy=False,
        )

        coarse = _coarse_from_f0(
            target_f0
        )

        frame_seconds = (
            float(
                self.window
            )
            / float(
                self.sr
            )
        )

        pad_seconds = float(
            getattr(
                self,
                "x_pad",
                0.0,
            )
            or 0.0
        )

        raw_duration = (
            float(
                len(
                    x
                )
            )
            / float(
                self.sr
            )
        )

        unpadded_duration = max(
            0.0,
            raw_duration
            - pad_seconds
            * 2.0,
        )

        try:
            _write_diagnostics(
                rmvpe_f0=rmvpe_source,
                reference_f0=reference_f0,
                reference_probability=probability,
                corrected_f0=corrected_source,
                target_f0=target_f0,
                weight=weight,
                summary=summary,
                frame_seconds=frame_seconds,
                pad_seconds=pad_seconds,
                unpadded_duration=unpadded_duration,
            )
        except Exception:
            print(
                "[F0 Stability Guard] diagnostic write failed.",
                file=sys.stderr,
            )
            traceback.print_exc()

        print(
            "[F0 Stability Guard] "
            f"mode={mode} / "
            f"corrected={summary.get('corrected_seconds', 0.0):.2f}s / "
            f"regions={summary.get('region_count', 0)} / "
            "normal RMVPE frames preserved"
        )

        for start, end in (
            summary.get(
                "regions",
                []
            )[
                :12
            ]
        ):
            # Region values still include upstream RVC padding.
            visible_start = max(
                0.0,
                float(
                    start
                )
                - pad_seconds,
            )
            visible_end = max(
                0.0,
                float(
                    end
                )
                - pad_seconds,
            )

            if (
                visible_end
                > visible_start
            ):
                print(
                    "[F0 Stability Guard] "
                    f"collapse correction candidate "
                    f"{visible_start:.2f}s ~ {visible_end:.2f}s"
                )

        return (
            coarse,
            target_f0,
        )

    stable_get_f0._vpa_f0_stability = True
    Pipeline.get_f0 = (
        stable_get_f0
    )


def main() -> int:
    install_f0_stability_patch()

    from infer.cli import main as upstream_main

    return int(
        upstream_main()
    )


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        raise SystemExit(
            130
        )
    except Exception as exc:
        print(
            "vpa-rvc-stable-cli: error: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
