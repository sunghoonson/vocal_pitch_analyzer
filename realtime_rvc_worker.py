from __future__ import annotations

# V39_REALTIME_RVC_VOICE_CHANGER_PATCH
#
# Runs under .venv_rvc and the pinned tools/rvc checkout.
# It keeps HuBERT + RMVPE + RVC synthesizer resident on GPU and processes
# fixed audio blocks with the same rolling-context + SOLA strategy used by
# the official RVC realtime implementation.

from pathlib import Path
import argparse
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
    return parser.parse_args()


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

        if self.f0_method not in {
            "rmvpe",
            "pm",
            "fcpe",
        }:
            raise ValueError(
                f"Unsupported f0 method: {self.f0_method}"
            )

        # Imports happen only after repo cwd/sys.path are prepared.
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

        return (
            result.detach()
            .float()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

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
            f"device={processor.config.device}",
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
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
