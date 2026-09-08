from __future__ import annotations

# V43_REALTIME_RVC_F0_STABILITY_GUARD_PATCH
# V39B_REALTIME_RVC_INDEX_HOTFIX
# V39_REALTIME_RVC_VOICE_CHANGER_PATCH
#
# Runs under .venv_rvc and the pinned tools/rvc checkout.
# It keeps HuBERT + RMVPE + RVC synthesizer resident on GPU and processes
# fixed audio blocks with the same rolling-context + SOLA strategy used by
# the official RVC realtime implementation.

from pathlib import Path
import argparse
import csv
import json
import math
import os
import socket
import struct
import sys
import time
import traceback

import numpy as np


def _recv_exact(
    sock: socket.socket,
    size: int,
) -> bytes:
    chunks: list[bytes] = []
    received = 0

    while received < int(
        size
    ):
        chunk = sock.recv(
            int(
                size
            )
            - received
        )

        if not chunk:
            raise ConnectionError(
                "client disconnected"
            )

        chunks.append(
            chunk
        )
        received += len(
            chunk
        )

    return b"".join(
        chunks
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--sample-rate", required=True, type=int)
    parser.add_argument("--pitch", default=0, type=int)
    parser.add_argument("--index-rate", default=0.35, type=float)
    parser.add_argument("--block-ms", default=200, type=int)
    parser.add_argument("--crossfade-ms", default=40, type=int)
    parser.add_argument("--extra-ms", default=1000, type=int)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--f0-guard", default=1, type=int)
    parser.add_argument("--f0-diagnostic", default=1, type=int)
    parser.add_argument("--rmvpe-threshold", default=0.05, type=float)
    parser.add_argument("--f0-quiet-rms-db", default=-48.0, type=float)
    parser.add_argument("--f0-min-voiced-ratio", default=0.15, type=float)
    parser.add_argument("--f0-max-gap-ms", default=30, type=int)
    parser.add_argument("--f0-min-run-ms", default=40, type=int)
    return parser.parse_args()


def _safe_dbfs(
    samples: np.ndarray,
) -> float:
    data = np.asarray(
        samples,
        dtype=np.float32,
    ).reshape(
        -1
    )

    if data.size <= 0:
        return -120.0

    rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    data.astype(
                        np.float64
                    )
                )
            )
            + 1e-12
        )
    )

    return float(
        20.0
        * np.log10(
            max(
                rms,
                1e-7,
            )
        )
    )


def _f0_stats(
    f0: np.ndarray,
) -> dict:
    data = np.asarray(
        f0,
        dtype=np.float32,
    ).reshape(
        -1
    )
    voiced = data[
        np.isfinite(
            data
        )
        & (
            data > 0.0
        )
    ]

    total = int(
        data.size
    )
    count = int(
        voiced.size
    )

    return {
        "frames": total,
        "voiced_frames": count,
        "voiced_ratio": (
            float(
                count
            )
            / float(
                total
            )
            if total > 0
            else 0.0
        ),
        "min": (
            float(
                np.min(
                    voiced
                )
            )
            if count > 0
            else 0.0
        ),
        "median": (
            float(
                np.median(
                    voiced
                )
            )
            if count > 0
            else 0.0
        ),
        "max": (
            float(
                np.max(
                    voiced
                )
            )
            if count > 0
            else 0.0
        ),
    }


def _stabilize_f0_array(
    raw_f0: np.ndarray,
    *,
    current_frames: int,
    input_rms_db: float,
    quiet_rms_db: float,
    min_voiced_ratio: float,
    max_gap_frames: int,
    min_run_frames: int,
) -> tuple[np.ndarray, dict]:
    """
    Conservative realtime F0 cleanup.

    Key difference from upstream realtime RVC:
      upstream np.interp() also extrapolates leading/trailing unvoiced frames,
      so one false positive can spread a pitch through a mostly silent region.

    Here:
      * invalid/negative F0 -> 0
      * very short voiced islands can be removed
      * only short INTERNAL gaps are interpolated
      * leading/trailing unvoiced regions stay 0
      * definitely quiet current blocks are forced unvoiced
    """

    source = np.asarray(
        raw_f0,
        dtype=np.float32,
    ).reshape(
        -1
    ).copy()

    source[
        ~np.isfinite(
            source
        )
    ] = 0.0
    source[
        source < 0.0
    ] = 0.0

    guarded = source.copy()
    total = int(
        guarded.size
    )

    current_count = max(
        1,
        min(
            int(
                current_frames
            ),
            total,
        ),
    ) if total > 0 else 0

    current_slice = slice(
        total
        - current_count,
        total,
    ) if current_count > 0 else slice(
        0,
        0,
    )

    raw_current = source[
        current_slice
    ]
    raw_current_stats = _f0_stats(
        raw_current
    )

    removed_run_frames = 0
    bridged_gap_frames = 0
    reasons: list[str] = []

    # Remove isolated short voiced runs.
    if (
        total > 0
        and int(
            min_run_frames
        ) > 1
    ):
        voiced = guarded > 0.0
        i = 0

        while i < total:
            if not voiced[
                i
            ]:
                i += 1
                continue

            start = i

            while (
                i < total
                and voiced[
                    i
                ]
            ):
                i += 1

            end = i
            run_length = (
                end
                - start
            )

            if run_length < int(
                min_run_frames
            ):
                guarded[
                    start:end
                ] = 0.0
                removed_run_frames += int(
                    run_length
                )

    if removed_run_frames > 0:
        reasons.append(
            "short_voiced_island"
        )

    # Only bridge INTERNAL unvoiced gaps. Never extrapolate through the
    # leading/trailing silence as upstream np.interp() does.
    max_gap = max(
        0,
        int(
            max_gap_frames
        ),
    )

    if (
        max_gap > 0
        and total >= 3
    ):
        i = 0

        while i < total:
            if guarded[
                i
            ] > 0.0:
                i += 1
                continue

            gap_start = i

            while (
                i < total
                and guarded[
                    i
                ] <= 0.0
            ):
                i += 1

            gap_end = i
            gap_length = (
                gap_end
                - gap_start
            )

            has_left = (
                gap_start > 0
                and guarded[
                    gap_start
                    - 1
                ] > 0.0
            )
            has_right = (
                gap_end < total
                and guarded[
                    gap_end
                ] > 0.0
            )

            if (
                has_left
                and has_right
                and gap_length <= max_gap
            ):
                left = float(
                    guarded[
                        gap_start
                        - 1
                    ]
                )
                right = float(
                    guarded[
                        gap_end
                    ]
                )

                guarded[
                    gap_start:gap_end
                ] = np.linspace(
                    left,
                    right,
                    gap_length
                    + 2,
                    dtype=np.float32,
                )[
                    1:-1
                ]
                bridged_gap_frames += int(
                    gap_length
                )

    if bridged_gap_frames > 0:
        reasons.append(
            "short_gap_bridge"
        )

    # Current block activity guard.
    clear_current = False

    if current_count > 0:
        if float(
            input_rms_db
        ) <= float(
            quiet_rms_db
        ):
            clear_current = True
            reasons.append(
                "quiet_block"
            )

        elif (
            float(
                input_rms_db
            )
            <= float(
                quiet_rms_db
            )
            + 6.0
            and float(
                raw_current_stats[
                    "voiced_ratio"
                ]
            )
            < float(
                min_voiced_ratio
            )
        ):
            clear_current = True
            reasons.append(
                "weak_sparse_f0"
            )

        if clear_current:
            guarded[
                current_slice
            ] = 0.0

    guarded_current = guarded[
        current_slice
    ]
    guarded_current_stats = _f0_stats(
        guarded_current
    )

    diagnostics = {
        "raw_current": raw_current_stats,
        "guarded_current": guarded_current_stats,
        "removed_run_frames": int(
            removed_run_frames
        ),
        "bridged_gap_frames": int(
            bridged_gap_frames
        ),
        "clear_current": bool(
            clear_current
        ),
        "reason": (
            "+".join(
                reasons
            )
            if reasons
            else "none"
        ),
    }

    return (
        guarded,
        diagnostics,
    )


class _SafeFaissSearchProxy:
    """
    Small adapter for RVC realtime index search.

    Official infer/rtrvc.py always asks FAISS for k=8 and then rejects the
    entire retrieval result if even one neighbor id is -1.

    RVC training writes IVF indices with nprobe=1. A sparse probed IVF list
    can contain fewer than 8 vectors, so FAISS legitimately pads the missing
    neighbors with -1. The upstream warning then misleadingly says the index
    is not an added index even when the selected file is already added_*.index.

    This proxy:
      * sanitizes non-finite HuBERT query values,
      * clamps k to ntotal for tiny indices,
      * retries IVF search with progressively larger nprobe if FAISS returns -1.
    """

    def __init__(
        self,
        index,
        *,
        faiss_module,
        log_prefix: str = "[Realtime RVC index]",
    ) -> None:
        self._index = index
        self._faiss = faiss_module
        self._log_prefix = str(log_prefix)
        self.ntotal = int(
            getattr(index, "ntotal", 0)
        )
        self.d = int(
            getattr(index, "d", 0)
        )
        self._ivf = None
        self._initial_nprobe = None
        self._negative_retry_count = 0

        try:
            self._ivf = self._faiss.extract_index_ivf(
                self._index
            )
            self._initial_nprobe = int(
                self._ivf.nprobe
            )

            nlist = max(
                1,
                int(
                    self._ivf.nlist
                ),
            )
            target = min(
                nlist,
                max(
                    8,
                    int(
                        self._ivf.nprobe
                    ),
                ),
            )
            self._ivf.nprobe = int(
                target
            )

            print(
                f"{self._log_prefix} IVF "
                f"ntotal={self.ntotal} / "
                f"nlist={nlist} / "
                f"nprobe={self._initial_nprobe}->{target}",
                flush=True,
            )
        except Exception:
            print(
                f"{self._log_prefix} Flat/non-IVF "
                f"ntotal={self.ntotal} / d={self.d}",
                flush=True,
            )

    def reconstruct_n(
        self,
        *args,
        **kwargs,
    ):
        return self._index.reconstruct_n(
            *args,
            **kwargs,
        )

    def search(
        self,
        query,
        k,
    ):
        q = np.asarray(
            query,
            dtype=np.float32,
        )

        if not np.isfinite(q).all():
            q = np.nan_to_num(
                q,
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        if self.ntotal <= 0:
            return self._index.search(
                q,
                int(k),
            )

        effective_k = max(
            1,
            min(
                int(k),
                self.ntotal,
            ),
        )

        score, ids = self._index.search(
            q,
            effective_k,
        )

        if (
            ids.size > 0
            and (ids < 0).any()
            and self._ivf is not None
        ):
            nlist = max(
                1,
                int(
                    self._ivf.nlist
                ),
            )
            current = max(
                1,
                int(
                    self._ivf.nprobe
                ),
            )

            # Grow only when necessary. Most models stop at nprobe=8.
            for candidate in (
                min(nlist, max(16, current * 2)),
                min(nlist, max(32, current * 4)),
                nlist,
            ):
                if candidate <= current:
                    continue

                self._ivf.nprobe = int(
                    candidate
                )
                score, ids = self._index.search(
                    q,
                    effective_k,
                )
                current = int(
                    candidate
                )

                if not (ids < 0).any():
                    self._negative_retry_count += 1

                    if (
                        self._negative_retry_count <= 3
                        or self._negative_retry_count % 100 == 0
                    ):
                        print(
                            f"{self._log_prefix} sparse IVF retry OK / "
                            f"nprobe={current}",
                            flush=True,
                        )
                    break

        return score, ids


class RealtimeProcessor:
    def __init__(
        self,
        *,
        repo: Path,
        model_path: Path,
        index_path: Path | None,
        sample_rate: int,
        pitch: int,
        index_rate: float,
        block_ms: int,
        crossfade_ms: int,
        extra_ms: int,
        f0_method: str,
        f0_guard_enabled: bool,
        f0_diagnostic_enabled: bool,
        rmvpe_threshold: float,
        f0_quiet_rms_db: float,
        f0_min_voiced_ratio: float,
        f0_max_gap_ms: int,
        f0_min_run_ms: int,
    ) -> None:
        self.repo = Path(
            repo
        ).resolve()
        self.sample_rate = max(
            8000,
            int(
                sample_rate
            ),
        )
        self.pitch = int(
            pitch
        )
        self.index_rate = float(
            index_rate
        )
        self.f0_method = str(
            f0_method
            or "rmvpe"
        ).strip().lower()

        self.f0_guard_enabled = bool(
            f0_guard_enabled
        )
        self.f0_diagnostic_enabled = bool(
            f0_diagnostic_enabled
        )
        self.rmvpe_threshold = max(
            0.001,
            min(
                float(
                    rmvpe_threshold
                ),
                0.99,
            ),
        )
        self.f0_quiet_rms_db = max(
            -90.0,
            min(
                float(
                    f0_quiet_rms_db
                ),
                -10.0,
            ),
        )
        self.f0_min_voiced_ratio = max(
            0.0,
            min(
                float(
                    f0_min_voiced_ratio
                ),
                1.0,
            ),
        )
        self.f0_max_gap_ms = max(
            0,
            min(
                int(
                    f0_max_gap_ms
                ),
                200,
            ),
        )
        self.f0_min_run_ms = max(
            0,
            min(
                int(
                    f0_min_run_ms
                ),
                200,
            ),
        )

        if self.f0_method not in {
            "rmvpe",
            "pm",
            "fcpe",
        }:
            raise ValueError(
                f"Unsupported f0 method: {self.f0_method}"
            )

        # Imports happen only after repo cwd/sys.path are prepared.
        import faiss
        import torch
        import torch.nn.functional as F
        import torchaudio.transforms as tat
        from configs.config import Config
        from infer.rtrvc import RVC

        self.torch = torch
        self.F = F
        self.tat = tat

        self.config = Config()

        effective_index = (
            str(
                index_path
            )
            if (
                index_path is not None
                and index_path.is_file()
                and self.index_rate > 0.0
            )
            else ""
        )

        effective_index_rate = (
            self.index_rate
            if effective_index
            else 0.0
        )

        self.rvc = RVC(
            self.pitch,
            0.0,
            str(
                model_path
            ),
            effective_index,
            float(
                effective_index_rate
            ),
            self.config,
            None,
        )

        # Official realtime RVC uses k=8 but training writes IVF nprobe=1.
        # Raise nprobe at runtime and retry sparse IVF searches without
        # modifying the user's .index file on disk.
        if (
            effective_index
            and float(
                effective_index_rate
            ) > 0.0
            and hasattr(
                self.rvc,
                "index",
            )
        ):
            original_index = self.rvc.index
            self.rvc.index = _SafeFaissSearchProxy(
                original_index,
                faiss_module=faiss,
            )

        if self.f0_method == "rmvpe":
            # Instance-level replacement. RVC.get_f0() still calls
            # self.get_f0_rmvpe(), but now the method preserves genuine
            # unvoiced regions and records the raw RMVPE result.
            self.rvc.get_f0_rmvpe = self._guarded_get_f0_rmvpe

        if not hasattr(
            self.rvc,
            "tgt_sr",
        ):
            raise RuntimeError(
                "RVC realtime model initialization failed: tgt_sr missing."
            )

        self.model_sample_rate = int(
            self.rvc.tgt_sr
        )

        self.zc = max(
            1,
            self.sample_rate
            // 100,
        )

        self.block_frame = (
            max(
                1,
                int(
                    round(
                        (
                            max(
                                100,
                                int(
                                    block_ms
                                ),
                            )
                            / 1000.0
                        )
                        * self.sample_rate
                        / self.zc
                    )
                ),
            )
            * self.zc
        )

        self.block_frame_16k = (
            160
            * self.block_frame
            // self.zc
        )

        self._f0_current_frames = max(
            1,
            int(
                round(
                    self.block_frame_16k
                    / 160.0
                )
            ),
        )
        self._f0_max_gap_frames = max(
            0,
            int(
                round(
                    self.f0_max_gap_ms
                    / 10.0
                )
            ),
        )
        self._f0_min_run_frames = max(
            0,
            int(
                math.ceil(
                    self.f0_min_run_ms
                    / 10.0
                )
            ),
        )

        self._f0_diag_path = (
            Path(
                __file__
            ).resolve().parent
            / "logs"
            / "realtime_rvc_f0_last.csv"
        )
        self._f0_summary_path = (
            Path(
                __file__
            ).resolve().parent
            / "logs"
            / "realtime_rvc_f0_last.json"
        )
        self._f0_diag_file = None
        self._f0_diag_writer = None
        self._f0_diag_last: dict = {}
        self._f0_diag_blocks = 0
        self._f0_guard_triggered_blocks = 0
        self._f0_quiet_blocks = 0
        self._f0_short_run_frames_removed = 0
        self._f0_gap_frames_bridged = 0
        self._f0_raw_voiced_ratio_sum = 0.0
        self._f0_guarded_voiced_ratio_sum = 0.0
        self._f0_previous_stable_median = 0.0
        self._f0_warmup = True

        if self.f0_diagnostic_enabled:
            self._open_f0_diagnostics()

        self.crossfade_frame = (
            max(
                1,
                int(
                    round(
                        (
                            max(
                                10,
                                int(
                                    crossfade_ms
                                ),
                            )
                            / 1000.0
                        )
                        * self.sample_rate
                        / self.zc
                    )
                ),
            )
            * self.zc
        )

        self.sola_buffer_frame = min(
            self.crossfade_frame,
            4
            * self.zc,
        )
        self.sola_search_frame = (
            self.zc
        )

        self.extra_frame = (
            max(
                1,
                int(
                    round(
                        (
                            max(
                                200,
                                int(
                                    extra_ms
                                ),
                            )
                            / 1000.0
                        )
                        * self.sample_rate
                        / self.zc
                    )
                ),
            )
            * self.zc
        )

        device = self.config.device

        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=device,
            dtype=torch.float32,
        )

        self.input_wav_res = torch.zeros(
            160
            * self.input_wav.shape[
                0
            ]
            // self.zc,
            device=device,
            dtype=torch.float32,
        )

        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )

        self.sola_den_kernel = torch.ones(
            1,
            1,
            self.sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )

        self.skip_head = (
            self.extra_frame
            // self.zc
        )
        self.return_length = (
            self.block_frame
            + self.sola_buffer_frame
            + self.sola_search_frame
        ) // self.zc

        phase = torch.linspace(
            0.0,
            1.0,
            steps=self.sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * phase
            )
            ** 2
        )
        self.fade_out_window = (
            1.0
            - self.fade_in_window
        )

        self.resampler = tat.Resample(
            orig_freq=self.sample_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(
            device
        )

        if (
            self.model_sample_rate
            != self.sample_rate
        ):
            self.resampler2 = tat.Resample(
                orig_freq=self.model_sample_rate,
                new_freq=self.sample_rate,
                dtype=torch.float32,
            ).to(
                device
            )
        else:
            self.resampler2 = None

        self.processed_blocks = 0
        self.last_infer_ms = 0.0

        self._warmup()

    def _open_f0_diagnostics(
        self,
    ) -> None:
        self._f0_diag_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._f0_diag_file = self._f0_diag_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )

        fields = [
            "block",
            "time_monotonic",
            "input_rms_db",
            "rmvpe_threshold",
            "raw_frames",
            "raw_voiced_frames",
            "raw_voiced_ratio",
            "raw_f0_min",
            "raw_f0_median",
            "raw_f0_max",
            "guarded_voiced_frames",
            "guarded_voiced_ratio",
            "guarded_f0_min",
            "guarded_f0_median",
            "guarded_f0_max",
            "removed_run_frames",
            "bridged_gap_frames",
            "clear_current",
            "reason",
            "previous_stable_median",
            "f0_jump_ratio",
            "output_rms_db",
            "infer_ms",
        ]

        self._f0_diag_writer = csv.DictWriter(
            self._f0_diag_file,
            fieldnames=fields,
        )
        self._f0_diag_writer.writeheader()
        self._f0_diag_file.flush()

    def _write_f0_diagnostic(
        self,
        *,
        output: np.ndarray,
        infer_ms: float,
    ) -> None:
        if (
            not self.f0_diagnostic_enabled
            or self._f0_diag_writer is None
            or self._f0_warmup
        ):
            return

        diag = dict(
            self._f0_diag_last
        )

        if not diag:
            return

        output_rms_db = _safe_dbfs(
            output
        )
        diag[
            "output_rms_db"
        ] = float(
            output_rms_db
        )
        diag[
            "infer_ms"
        ] = float(
            infer_ms
        )

        self._f0_diag_writer.writerow(
            diag
        )
        self._f0_diag_blocks += 1
        self._f0_raw_voiced_ratio_sum += float(
            diag.get(
                "raw_voiced_ratio",
                0.0,
            )
        )
        self._f0_guarded_voiced_ratio_sum += float(
            diag.get(
                "guarded_voiced_ratio",
                0.0,
            )
        )
        self._f0_short_run_frames_removed += int(
            diag.get(
                "removed_run_frames",
                0,
            )
        )
        self._f0_gap_frames_bridged += int(
            diag.get(
                "bridged_gap_frames",
                0,
            )
        )

        if bool(
            diag.get(
                "clear_current",
                False,
            )
        ):
            self._f0_guard_triggered_blocks += 1

        if "quiet_block" in str(
            diag.get(
                "reason",
                "",
            )
        ):
            self._f0_quiet_blocks += 1

        if (
            self._f0_diag_blocks <= 5
            or self._f0_diag_blocks
            % 10
            == 0
        ):
            self._f0_diag_file.flush()

    def _guarded_get_f0_rmvpe(
        self,
        x,
        f0_up_key,
    ):
        if not hasattr(
            self.rvc,
            "model_rmvpe",
        ):
            from infer.rmvpe import RMVPE

            print(
                "[F0 Guard] RMVPE 모델 불러오는 중 / "
                f"threshold={self.rmvpe_threshold:.3f}",
                flush=True,
            )
            self.rvc.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=self.rvc.is_half,
                device=self.rvc.device,
            )

        raw_f0 = self.rvc.model_rmvpe.infer_from_audio(
            x,
            thred=float(
                self.rmvpe_threshold
            ),
        )
        raw_f0 = np.asarray(
            raw_f0,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if hasattr(
            x,
            "detach",
        ):
            x_np = (
                x.detach()
                .float()
                .cpu()
                .numpy()
                .reshape(
                    -1
                )
            )
        else:
            x_np = np.asarray(
                x,
                dtype=np.float32,
            ).reshape(
                -1
            )

        latest_audio = x_np[
            -min(
                x_np.size,
                max(
                    1,
                    int(
                        self.block_frame_16k
                    ),
                ),
            ):
        ]
        input_rms_db = _safe_dbfs(
            latest_audio
        )

        raw_current = raw_f0[
            -min(
                raw_f0.size,
                self._f0_current_frames,
            ):
        ]
        raw_current_stats = _f0_stats(
            raw_current
        )

        if self.f0_guard_enabled:
            guarded_f0, info = _stabilize_f0_array(
                raw_f0,
                current_frames=self._f0_current_frames,
                input_rms_db=input_rms_db,
                quiet_rms_db=self.f0_quiet_rms_db,
                min_voiced_ratio=self.f0_min_voiced_ratio,
                max_gap_frames=self._f0_max_gap_frames,
                min_run_frames=self._f0_min_run_frames,
            )

        else:
            # Exact upstream behavior for comparison:
            # all zero/unvoiced positions are interpolated/extrapolated from
            # any detected voiced frames.
            guarded_f0 = raw_f0.copy()
            uv = (
                guarded_f0
                == 0.0
            )

            if np.any(
                ~uv
            ):
                guarded_f0[
                    uv
                ] = np.interp(
                    np.where(
                        uv
                    )[
                        0
                    ],
                    np.where(
                        ~uv
                    )[
                        0
                    ],
                    guarded_f0[
                        ~uv
                    ],
                )

            guarded_current_stats = _f0_stats(
                guarded_f0[
                    -min(
                        guarded_f0.size,
                        self._f0_current_frames,
                    ):
                ]
            )
            info = {
                "raw_current": raw_current_stats,
                "guarded_current": guarded_current_stats,
                "removed_run_frames": 0,
                "bridged_gap_frames": int(
                    np.sum(
                        uv
                    )
                )
                if np.any(
                    ~uv
                )
                else 0,
                "clear_current": False,
                "reason": "guard_off_upstream_interp",
            }

        guarded_current_stats = info[
            "guarded_current"
        ]

        previous_median = float(
            self._f0_previous_stable_median
        )
        current_median = float(
            guarded_current_stats[
                "median"
            ]
        )
        jump_ratio = 0.0

        if (
            previous_median > 0.0
            and current_median > 0.0
        ):
            jump_ratio = (
                current_median
                / previous_median
            )

        if (
            current_median > 0.0
            and float(
                guarded_current_stats[
                    "voiced_ratio"
                ]
            )
            >= float(
                self.f0_min_voiced_ratio
            )
            and input_rms_db
            > float(
                self.f0_quiet_rms_db
            )
        ):
            self._f0_previous_stable_median = (
                current_median
            )

        shifted_f0 = (
            guarded_f0
            * pow(
                2.0,
                float(
                    f0_up_key
                )
                / 12.0,
            )
        )

        block_number = int(
            self.processed_blocks
            + 1
        )

        self._f0_diag_last = {
            "block": block_number,
            "time_monotonic": float(
                time.monotonic()
            ),
            "input_rms_db": float(
                input_rms_db
            ),
            "rmvpe_threshold": float(
                self.rmvpe_threshold
            ),
            "raw_frames": int(
                raw_current_stats[
                    "frames"
                ]
            ),
            "raw_voiced_frames": int(
                raw_current_stats[
                    "voiced_frames"
                ]
            ),
            "raw_voiced_ratio": float(
                raw_current_stats[
                    "voiced_ratio"
                ]
            ),
            "raw_f0_min": float(
                raw_current_stats[
                    "min"
                ]
            ),
            "raw_f0_median": float(
                raw_current_stats[
                    "median"
                ]
            ),
            "raw_f0_max": float(
                raw_current_stats[
                    "max"
                ]
            ),
            "guarded_voiced_frames": int(
                guarded_current_stats[
                    "voiced_frames"
                ]
            ),
            "guarded_voiced_ratio": float(
                guarded_current_stats[
                    "voiced_ratio"
                ]
            ),
            "guarded_f0_min": float(
                guarded_current_stats[
                    "min"
                ]
            ),
            "guarded_f0_median": float(
                guarded_current_stats[
                    "median"
                ]
            ),
            "guarded_f0_max": float(
                guarded_current_stats[
                    "max"
                ]
            ),
            "removed_run_frames": int(
                info[
                    "removed_run_frames"
                ]
            ),
            "bridged_gap_frames": int(
                info[
                    "bridged_gap_frames"
                ]
            ),
            "clear_current": bool(
                info[
                    "clear_current"
                ]
            ),
            "reason": str(
                info[
                    "reason"
                ]
            ),
            "previous_stable_median": float(
                previous_median
            ),
            "f0_jump_ratio": float(
                jump_ratio
            ),
            "output_rms_db": 0.0,
            "infer_ms": 0.0,
        }

        # Console diagnostic is rate-limited. The CSV has every block.
        reason = str(
            info[
                "reason"
            ]
        )

        if (
            not self._f0_warmup
            and (
                block_number <= 5
                or bool(
                    info[
                        "clear_current"
                    ]
                )
                or block_number
                % 50
                == 0
            )
        ):
            print(
                "[F0 Guard] "
                f"block={block_number} / "
                f"rms={input_rms_db:.1f}dBFS / "
                f"raw={raw_current_stats['voiced_ratio']:.2f} "
                f"{raw_current_stats['median']:.1f}Hz / "
                f"guarded={guarded_current_stats['voiced_ratio']:.2f} "
                f"{guarded_current_stats['median']:.1f}Hz / "
                f"reason={reason}",
                flush=True,
            )

        return self.rvc.get_f0_post(
            shifted_f0
        )

    def close(
        self,
    ) -> None:
        if self.f0_diagnostic_enabled:
            blocks = max(
                1,
                int(
                    self._f0_diag_blocks
                ),
            )

            summary = {
                "version": "v4.3",
                "f0_guard_enabled": bool(
                    self.f0_guard_enabled
                ),
                "f0_diagnostic_enabled": bool(
                    self.f0_diagnostic_enabled
                ),
                "rmvpe_threshold": float(
                    self.rmvpe_threshold
                ),
                "quiet_rms_db": float(
                    self.f0_quiet_rms_db
                ),
                "min_voiced_ratio": float(
                    self.f0_min_voiced_ratio
                ),
                "max_gap_ms": int(
                    self.f0_max_gap_ms
                ),
                "min_run_ms": int(
                    self.f0_min_run_ms
                ),
                "blocks": int(
                    self._f0_diag_blocks
                ),
                "guard_triggered_blocks": int(
                    self._f0_guard_triggered_blocks
                ),
                "quiet_blocks": int(
                    self._f0_quiet_blocks
                ),
                "short_run_frames_removed": int(
                    self._f0_short_run_frames_removed
                ),
                "gap_frames_bridged": int(
                    self._f0_gap_frames_bridged
                ),
                "mean_raw_voiced_ratio": float(
                    self._f0_raw_voiced_ratio_sum
                    / blocks
                ),
                "mean_guarded_voiced_ratio": float(
                    self._f0_guarded_voiced_ratio_sum
                    / blocks
                ),
                "csv": str(
                    self._f0_diag_path
                ),
            }

            try:
                self._f0_summary_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                self._f0_summary_path.write_text(
                    json.dumps(
                        summary,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        if self._f0_diag_file is not None:
            try:
                self._f0_diag_file.flush()
            except Exception:
                pass

            try:
                self._f0_diag_file.close()
            except Exception:
                pass

            self._f0_diag_file = None
            self._f0_diag_writer = None

    def _warmup(
        self,
    ) -> None:
        # Avoid a very loud or stale first block. One low-level sine probe
        # warms RMVPE/HuBERT/synthesizer then every rolling cache is cleared.
        torch = self.torch

        try:
            count = int(
                self.block_frame
            )
            phase = torch.arange(
                count,
                device=self.config.device,
                dtype=torch.float32,
            )
            probe = (
                0.01
                * torch.sin(
                    2.0
                    * np.pi
                    * 220.0
                    * phase
                    / float(
                        self.sample_rate
                    )
                )
            )
            self._process_tensor_block(
                probe
            )
        except Exception:
            print(
                "[Realtime RVC] warmup warning:",
                traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
        finally:
            self.input_wav.zero_()
            self.input_wav_res.zero_()
            self.sola_buffer.zero_()

            if hasattr(
                self.rvc,
                "cache_pitch",
            ):
                self.rvc.cache_pitch.zero_()

            if hasattr(
                self.rvc,
                "cache_pitchf",
            ):
                self.rvc.cache_pitchf.zero_()

            self._f0_diag_last = {}
            self._f0_warmup = False

    def process(
        self,
        block: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch

        data = np.asarray(
            block,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if data.size < self.block_frame:
            data = np.pad(
                data,
                (
                    0,
                    self.block_frame
                    - data.size,
                ),
            )
        elif data.size > self.block_frame:
            data = data[
                : self.block_frame
            ]

        tensor = torch.from_numpy(
            data
        ).to(
            self.config.device,
            dtype=torch.float32,
        )

        started = time.perf_counter()
        result = self._process_tensor_block(
            tensor
        )
        self.last_infer_ms = (
            time.perf_counter()
            - started
        ) * 1000.0
        self.processed_blocks += 1

        output = (
            result.detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        self._write_f0_diagnostic(
            output=output,
            infer_ms=self.last_infer_ms,
        )

        return output

    def _process_tensor_block(
        self,
        block,
    ):
        torch = self.torch
        F = self.F

        # Rolling source context.
        self.input_wav[
            : -self.block_frame
        ] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[
            -self.block_frame :
        ] = block

        # Rolling 16 kHz context.
        self.input_wav_res[
            : -self.block_frame_16k
        ] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()

        resample_input = self.input_wav[
            -self.block_frame
            - 2
            * self.zc :
        ]

        resampled = self.resampler(
            resample_input
        )

        tail_count = (
            160
            * (
                self.block_frame
                // self.zc
                + 1
            )
        )

        tail = resampled[
            160:
        ]

        if tail.numel() < tail_count:
            tail = F.pad(
                tail,
                (
                    tail_count
                    - tail.numel(),
                    0,
                ),
            )
        elif tail.numel() > tail_count:
            tail = tail[
                -tail_count:
            ]

        self.input_wav_res[
            -tail_count:
        ] = tail

        inferred = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.f0_method,
        )

        if self.resampler2 is not None:
            inferred = self.resampler2(
                inferred
            )

        inferred = inferred.reshape(
            -1
        ).float()

        required = (
            self.block_frame
            + self.sola_buffer_frame
            + self.sola_search_frame
        )

        if inferred.numel() < required:
            inferred = F.pad(
                inferred,
                (
                    0,
                    required
                    - inferred.numel(),
                ),
            )

        # SOLA alignment / crossfade, following the official realtime GUI.
        if self.sola_buffer_frame > 0:
            conv_input = inferred[
                None,
                None,
                :
                self.sola_buffer_frame
                + self.sola_search_frame
            ]

            cor_nom = F.conv1d(
                conv_input,
                self.sola_buffer[
                    None,
                    None,
                    :
                ],
            )
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input
                    ** 2,
                    self.sola_den_kernel,
                )
                + 1e-8
            )
            ratio = (
                cor_nom[
                    0,
                    0
                ]
                / cor_den[
                    0,
                    0
                ]
            )
            sola_offset = int(
                torch.argmax(
                    ratio
                ).item()
            )

            inferred = inferred[
                sola_offset:
            ]

            required_after = (
                self.block_frame
                + self.sola_buffer_frame
            )

            if inferred.numel() < required_after:
                inferred = F.pad(
                    inferred,
                    (
                        0,
                        required_after
                        - inferred.numel(),
                    ),
                )

            inferred[
                : self.sola_buffer_frame
            ] *= self.fade_in_window
            inferred[
                : self.sola_buffer_frame
            ] += (
                self.sola_buffer
                * self.fade_out_window
            )

            self.sola_buffer[:] = inferred[
                self.block_frame :
                self.block_frame
                + self.sola_buffer_frame
            ]

        output = inferred[
            : self.block_frame
        ]

        if output.numel() < self.block_frame:
            output = F.pad(
                output,
                (
                    0,
                    self.block_frame
                    - output.numel(),
                ),
            )

        return output


def main() -> int:
    args = _parse_args()

    repo = Path(
        args.repo
    ).expanduser().resolve()
    model = Path(
        args.model
    ).expanduser().resolve()
    index = (
        Path(
            args.index
        ).expanduser().resolve()
        if str(
            args.index
        ).strip()
        else None
    )

    if not repo.is_dir():
        raise FileNotFoundError(
            repo
        )

    if not model.is_file():
        raise FileNotFoundError(
            model
        )

    # Config imports expect the RVC project root as cwd and import root.
    os.chdir(
        repo
    )
    sys.path.insert(
        0,
        str(
            repo
        ),
    )

    os.environ.setdefault(
        "OPENBLAS_NUM_THREADS",
        "1",
    )
    os.environ.setdefault(
        "weight_root",
        str(
            repo
            / "assets"
            / "weights"
        ),
    )
    os.environ.setdefault(
        "index_root",
        str(
            repo
            / "logs"
        ),
    )
    os.environ.setdefault(
        "outside_index_root",
        str(
            repo
            / "assets"
            / "indices"
        ),
    )
    os.environ.setdefault(
        "rmvpe_root",
        str(
            repo
            / "assets"
            / "rmvpe"
        ),
    )

    # Parse worker args before Config sees sys.argv.
    sys.argv = [
        sys.argv[
            0
        ]
    ]

    try:
        processor = RealtimeProcessor(
            repo=repo,
            model_path=model,
            index_path=index,
            sample_rate=int(
                args.sample_rate
            ),
            pitch=int(
                args.pitch
            ),
            index_rate=float(
                args.index_rate
            ),
            block_ms=int(
                args.block_ms
            ),
            crossfade_ms=int(
                args.crossfade_ms
            ),
            extra_ms=int(
                args.extra_ms
            ),
            f0_method=str(
                args.f0_method
            ),
            f0_guard_enabled=bool(
                int(
                    args.f0_guard
                )
            ),
            f0_diagnostic_enabled=bool(
                int(
                    args.f0_diagnostic
                )
            ),
            rmvpe_threshold=float(
                args.rmvpe_threshold
            ),
            f0_quiet_rms_db=float(
                args.f0_quiet_rms_db
            ),
            f0_min_voiced_ratio=float(
                args.f0_min_voiced_ratio
            ),
            f0_max_gap_ms=int(
                args.f0_max_gap_ms
            ),
            f0_min_run_ms=int(
                args.f0_min_run_ms
            ),
        )

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        server.bind(
            (
                "127.0.0.1",
                int(
                    args.port
                ),
            )
        )
        server.listen(
            1
        )

        print(
            "VPA_REALTIME_RVC_READY|"
            f"port={int(args.port)}|"
            f"sr={processor.sample_rate}|"
            f"model_sr={processor.model_sample_rate}|"
            f"block_frames={processor.block_frame}|"
            f"block_ms={int(args.block_ms)}|"
            f"device={processor.config.device}|"
            f"f0_guard={1 if processor.f0_guard_enabled else 0}|"
            f"rmvpe_threshold={processor.rmvpe_threshold:.3f}",
            flush=True,
        )

        client, address = server.accept()
        print(
            "[Realtime RVC] client connected: "
            f"{address}",
            file=sys.stderr,
            flush=True,
        )

        with client:
            while True:
                header = _recv_exact(
                    client,
                    4,
                )
                payload_size = struct.unpack(
                    "<I",
                    header,
                )[0]

                if payload_size == 0:
                    break

                if (
                    payload_size
                    > 64
                    * 1024
                    * 1024
                    or payload_size
                    % 4
                    != 0
                ):
                    raise ValueError(
                        "invalid block size: "
                        f"{payload_size}"
                    )

                payload = _recv_exact(
                    client,
                    int(
                        payload_size
                    ),
                )
                block = np.frombuffer(
                    payload,
                    dtype="<f4",
                ).astype(
                    np.float32,
                    copy=True,
                )

                output = processor.process(
                    block
                )
                response = (
                    np.asarray(
                        output,
                        dtype="<f4",
                    )
                    .reshape(
                        -1
                    )
                    .tobytes()
                )

                client.sendall(
                    struct.pack(
                        "<I",
                        len(
                            response
                        ),
                    )
                    + response
                )

                if (
                    processor.processed_blocks
                    <= 3
                    or processor.processed_blocks
                    % 50
                    == 0
                ):
                    print(
                        "[Realtime RVC] "
                        f"block={processor.processed_blocks} / "
                        f"infer={processor.last_infer_ms:.1f}ms",
                        file=sys.stderr,
                        flush=True,
                    )

        server.close()
        processor.close()
        return 0

    except Exception as exc:
        message = (
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "VPA_REALTIME_RVC_ERROR|"
            + message,
            flush=True,
        )
        traceback.print_exc(
            file=sys.stderr
        )

        try:
            processor.close()
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
