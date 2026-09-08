from __future__ import annotations

# V39_REALTIME_RVC_VOICE_CHANGER_PATCH
#
# Main-process controller for the persistent RVC realtime worker.
#
# The heavy Torch/HuBERT/RMVPE/RVC model remains in .venv_rvc.
# The PySide6 process only sends/receives Float32 blocks through localhost.

from pathlib import Path
from typing import Callable
import os
import queue
import socket
import struct
import subprocess
import threading
import time

import numpy as np


LogCallback = Callable[[str], None]
AudioCallback = Callable[[np.ndarray], None]


class RealtimeRVCError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parent


def rvc_repo_dir() -> Path:
    return project_root() / "tools" / "rvc"


def rvc_python() -> Path:
    if os.name == "nt":
        return (
            project_root()
            / ".venv_rvc"
            / "Scripts"
            / "python.exe"
        )

    return (
        project_root()
        / ".venv_rvc"
        / "bin"
        / "python"
    )


def worker_path() -> Path:
    return project_root() / "realtime_rvc_worker.py"


def realtime_rvc_status_text() -> str:
    py = rvc_python()
    repo = rvc_repo_dir()
    worker = worker_path()

    missing: list[str] = []

    if not py.is_file():
        missing.append(".venv_rvc")

    if not (repo / "infer" / "rtrvc.py").is_file():
        missing.append("tools/rvc/infer/rtrvc.py")

    if not (repo / "assets" / "rmvpe" / "rmvpe.pt").is_file():
        missing.append("RMVPE")

    if not (
        repo
        / "assets"
        / "hubert_base"
        / "pytorch_model.bin"
    ).is_file():
        missing.append("HuBERT")

    if not worker.is_file():
        missing.append("realtime_rvc_worker.py")

    if missing:
        return (
            "Realtime RVC 준비 안 됨: "
            + ", ".join(missing)
            + " / CHECK_REALTIME_RVC.bat 확인"
        )

    return (
        "Realtime RVC runtime OK / "
        ".venv_rvc + pinned tools/rvc 사용"
    )


def _free_local_port() -> int:
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:
        sock.bind(
            ("127.0.0.1", 0)
        )
        return int(
            sock.getsockname()[1]
        )


def _recv_exact(
    sock: socket.socket,
    size: int,
) -> bytes:
    target = max(
        0,
        int(size),
    )
    chunks: list[bytes] = []
    received = 0

    while received < target:
        chunk = sock.recv(
            target - received
        )

        if not chunk:
            raise ConnectionError(
                "Realtime RVC worker socket closed."
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


class RealtimeRVCClient:
    def __init__(
        self,
        *,
        log_callback: LogCallback | None = None,
        audio_callback: AudioCallback | None = None,
    ) -> None:
        self.log_callback = log_callback
        self.audio_callback = audio_callback

        self.process: subprocess.Popen | None = None
        self.socket: socket.socket | None = None

        self.sample_rate = 48000
        self.block_ms = 200
        self.block_frames = 9600

        self._input_queue: queue.Queue[np.ndarray] = (
            queue.Queue(
                maxsize=64
            )
        )
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

        self._error = ""
        self._ready_info = ""
        self._last_roundtrip_ms = 0.0
        self._processed_blocks = 0
        self._dropped_samples = 0
        self._sent_samples = 0
        self._received_samples = 0

    def _log(
        self,
        text: str,
    ) -> None:
        clean = str(
            text
        ).strip()

        if (
            clean
            and self.log_callback
            is not None
        ):
            self.log_callback(
                clean
            )

    @property
    def ready(self) -> bool:
        return (
            self._ready.is_set()
            and not self._failed.is_set()
            and self.socket is not None
        )

    @property
    def error(self) -> str:
        with self._state_lock:
            return str(
                self._error
            )

    def _set_error(
        self,
        message: str,
    ) -> None:
        with self._state_lock:
            self._error = str(
                message
            )

        self._failed.set()
        self._ready.set()
        self._log(
            "[Realtime RVC] ERROR: "
            + str(
                message
            )
        )

    def _pipe_reader(
        self,
        pipe,
        *,
        is_stdout: bool,
    ) -> None:
        if pipe is None:
            return

        try:
            for raw in iter(
                pipe.readline,
                "",
            ):
                line = str(
                    raw
                ).strip()

                if not line:
                    continue

                if (
                    is_stdout
                    and line.startswith(
                        "VPA_REALTIME_RVC_READY|"
                    )
                ):
                    with self._state_lock:
                        self._ready_info = line

                    self._ready.set()

                elif (
                    is_stdout
                    and line.startswith(
                        "VPA_REALTIME_RVC_ERROR|"
                    )
                ):
                    self._set_error(
                        line.split(
                            "|",
                            1,
                        )[1]
                    )

                else:
                    prefix = (
                        "[RVC worker] "
                        if is_stdout
                        else "[RVC worker stderr] "
                    )
                    self._log(
                        prefix
                        + line
                    )
        except Exception as exc:
            if not self._stop.is_set():
                self._log(
                    "[Realtime RVC] log reader ended: "
                    f"{type(exc).__name__}: {exc}"
                )

    def start(
        self,
        *,
        model_path: str | Path,
        index_path: str | Path | None,
        sample_rate: int,
        pitch: int = 0,
        index_rate: float = 0.35,
        block_ms: int = 200,
        crossfade_ms: int = 40,
        extra_ms: int = 1000,
        f0_method: str = "rmvpe",
        timeout: float = 120.0,
    ) -> None:
        self.stop()

        model = Path(
            model_path
        ).expanduser().resolve()

        if not model.is_file():
            raise FileNotFoundError(
                model
            )

        index = (
            Path(
                index_path
            ).expanduser().resolve()
            if index_path
            else None
        )

        effective_index_rate = float(
            index_rate
        )

        if (
            index is None
            or not index.is_file()
        ):
            index = None
            effective_index_rate = 0.0

        py = rvc_python()
        repo = rvc_repo_dir()
        worker = worker_path()

        if not py.is_file():
            raise RealtimeRVCError(
                f"RVC Python not found: {py}"
            )

        if not (
            repo
            / "infer"
            / "rtrvc.py"
        ).is_file():
            raise RealtimeRVCError(
                "Pinned RVC realtime backend not found: "
                f"{repo / 'infer' / 'rtrvc.py'}"
            )

        if not worker.is_file():
            raise RealtimeRVCError(
                f"Worker not found: {worker}"
            )

        self.sample_rate = max(
            8000,
            int(
                sample_rate
            ),
        )
        self.block_ms = max(
            100,
            min(
                int(
                    block_ms
                ),
                1000,
            ),
        )

        zc = max(
            1,
            self.sample_rate
            // 100,
        )
        self.block_frames = (
            max(
                1,
                int(
                    round(
                        (
                            self.block_ms
                            / 1000.0
                        )
                        * self.sample_rate
                        / zc
                    )
                ),
            )
            * zc
        )

        self._stop.clear()
        self._ready.clear()
        self._failed.clear()

        with self._state_lock:
            self._error = ""
            self._ready_info = ""

        while True:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

        port = _free_local_port()

        command = [
            str(
                py
            ),
            "-u",
            str(
                worker
            ),
            "--repo",
            str(
                repo
            ),
            "--port",
            str(
                port
            ),
            "--model",
            str(
                model
            ),
            "--sample-rate",
            str(
                self.sample_rate
            ),
            "--pitch",
            str(
                int(
                    pitch
                )
            ),
            "--index-rate",
            f"{effective_index_rate:.6f}",
            "--block-ms",
            str(
                int(
                    self.block_ms
                )
            ),
            "--crossfade-ms",
            str(
                int(
                    crossfade_ms
                )
            ),
            "--extra-ms",
            str(
                int(
                    extra_ms
                )
            ),
            "--f0-method",
            str(
                f0_method
                or "rmvpe"
            ),
        ]

        if index is not None:
            command.extend(
                [
                    "--index",
                    str(
                        index
                    ),
                ]
            )

        env = os.environ.copy()
        env.setdefault(
            "PYTHONUTF8",
            "1",
        )
        env.setdefault(
            "PYTHONIOENCODING",
            "utf-8",
        )
        env.setdefault(
            "OPENBLAS_NUM_THREADS",
            "1",
        )

        # Eager mode is the conservative first integration.
        env.setdefault(
            "RVC_CUDA_GRAPH",
            "0",
        )

        self._log(
            "[Realtime RVC] worker 시작 / "
            f"model={model.name} / "
            f"pitch={int(pitch):+d} / "
            f"index_rate={effective_index_rate:.2f} / "
            f"block={self.block_ms}ms / "
            f"sr={self.sample_rate}"
        )

        creationflags = int(
            getattr(
                subprocess,
                "CREATE_NO_WINDOW",
                0,
            )
        )

        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(
                    repo
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=env,
            )
        except OSError as exc:
            self.process = None
            raise RealtimeRVCError(
                f"worker launch failed: {exc}"
            ) from exc

        self._stdout_thread = threading.Thread(
            target=self._pipe_reader,
            args=(
                self.process.stdout,
            ),
            kwargs={
                "is_stdout": True,
            },
            name="RealtimeRVCStdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pipe_reader,
            args=(
                self.process.stderr,
            ),
            kwargs={
                "is_stdout": False,
            },
            name="RealtimeRVCStderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        deadline = time.monotonic() + max(
            5.0,
            float(
                timeout
            ),
        )

        while (
            not self._ready.wait(
                timeout=0.05
            )
        ):
            if (
                self.process.poll()
                is not None
            ):
                raise RealtimeRVCError(
                    "Realtime RVC worker exited during model load. "
                    f"exit={self.process.returncode}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Realtime RVC model load timeout."
                )

        if self._failed.is_set():
            raise RealtimeRVCError(
                self.error
                or "Realtime RVC worker failed."
            )

        try:
            sock = socket.create_connection(
                (
                    "127.0.0.1",
                    int(
                        port
                    ),
                ),
                timeout=10.0,
            )
            sock.settimeout(
                30.0
            )
            self.socket = sock

        except Exception:
            self.stop()
            raise

        self._thread = threading.Thread(
            target=self._processing_loop,
            name="RealtimeRVCClient",
            daemon=True,
        )
        self._thread.start()

        self._log(
            "[Realtime RVC] READY / "
            + (
                self._ready_info
                or "worker connected"
            )
        )

    def push(
        self,
        samples: np.ndarray,
    ) -> None:
        if not self.ready:
            return

        data = np.asarray(
            samples,
            dtype=np.float32,
        ).reshape(
            -1
        ).copy()

        if data.size <= 0:
            return

        # Keep latency bounded. If the worker falls behind, old speech is
        # less useful than the newest speech for a game voice changer.
        while self._input_queue.qsize() > 24:
            try:
                old = self._input_queue.get_nowait()
            except queue.Empty:
                break

            self._dropped_samples += int(
                old.size
            )

        try:
            self._input_queue.put_nowait(
                data
            )
        except queue.Full:
            try:
                old = self._input_queue.get_nowait()
                self._dropped_samples += int(
                    old.size
                )
            except queue.Empty:
                pass

            try:
                self._input_queue.put_nowait(
                    data
                )
            except queue.Full:
                self._dropped_samples += int(
                    data.size
                )

    def _processing_loop(
        self,
    ) -> None:
        accumulator = np.zeros(
            0,
            dtype=np.float32,
        )

        try:
            while not self._stop.is_set():
                try:
                    chunk = self._input_queue.get(
                        timeout=0.10
                    )
                except queue.Empty:
                    continue

                if chunk is None:
                    break

                accumulator = np.concatenate(
                    (
                        accumulator,
                        np.asarray(
                            chunk,
                            dtype=np.float32,
                        ).reshape(
                            -1
                        ),
                    )
                )

                while (
                    accumulator.size
                    >= self.block_frames
                    and not self._stop.is_set()
                ):
                    block = accumulator[
                        : self.block_frames
                    ].copy()
                    accumulator = accumulator[
                        self.block_frames :
                    ]

                    started = time.perf_counter()
                    output = self._exchange_block(
                        block
                    )
                    elapsed = (
                        time.perf_counter()
                        - started
                    ) * 1000.0

                    with self._state_lock:
                        self._last_roundtrip_ms = float(
                            elapsed
                        )
                        self._processed_blocks += 1
                        self._sent_samples += int(
                            block.size
                        )
                        self._received_samples += int(
                            output.size
                        )

                    if self.audio_callback is not None:
                        self.audio_callback(
                            output
                        )

        except Exception as exc:
            if not self._stop.is_set():
                self._set_error(
                    f"{type(exc).__name__}: {exc}"
                )

    def _exchange_block(
        self,
        block: np.ndarray,
    ) -> np.ndarray:
        sock = self.socket

        if sock is None:
            raise ConnectionError(
                "Realtime RVC socket is not connected."
            )

        payload = (
            np.asarray(
                block,
                dtype="<f4",
            )
            .reshape(
                -1
            )
            .tobytes()
        )

        sock.sendall(
            struct.pack(
                "<I",
                len(
                    payload
                ),
            )
            + payload
        )

        response_size = struct.unpack(
            "<I",
            _recv_exact(
                sock,
                4,
            ),
        )[0]

        if (
            response_size <= 0
            or response_size
            > 64 * 1024 * 1024
        ):
            raise RealtimeRVCError(
                "Invalid worker response size: "
                f"{response_size}"
            )

        response = _recv_exact(
            sock,
            int(
                response_size
            ),
        )

        output = np.frombuffer(
            response,
            dtype="<f4",
        ).astype(
            np.float32,
            copy=True,
        )

        if output.size < self.block_frames:
            output = np.pad(
                output,
                (
                    0,
                    self.block_frames
                    - output.size,
                ),
            )
        elif output.size > self.block_frames:
            output = output[
                : self.block_frames
            ]

        return output

    def stop(self) -> None:
        self._stop.set()

        sock = self.socket
        self.socket = None

        if sock is not None:
            try:
                sock.sendall(
                    struct.pack(
                        "<I",
                        0,
                    )
                )
            except Exception:
                pass

            try:
                sock.shutdown(
                    socket.SHUT_RDWR
                )
            except Exception:
                pass

            try:
                sock.close()
            except Exception:
                pass

        if (
            self._thread is not None
            and self._thread.is_alive()
            and self._thread
            is not threading.current_thread()
        ):
            self._thread.join(
                timeout=2.0
            )

        self._thread = None

        process = self.process
        self.process = None

        if process is not None:
            try:
                process.wait(
                    timeout=3.0
                )
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

                try:
                    process.wait(
                        timeout=2.0
                    )
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass

        self._ready.clear()

        while True:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

    def snapshot(self) -> dict:
        with self._state_lock:
            last_ms = float(
                self._last_roundtrip_ms
            )
            blocks = int(
                self._processed_blocks
            )
            sent = int(
                self._sent_samples
            )
            received = int(
                self._received_samples
            )
            error = str(
                self._error
            )

        queue_packets = int(
            self._input_queue.qsize()
        )
        queue_ms = (
            queue_packets
            * 20.0
        )

        return {
            "ready": bool(
                self.ready
            ),
            "error": error,
            "last_roundtrip_ms": last_ms,
            "processed_blocks": blocks,
            "sent_samples": sent,
            "received_samples": received,
            "dropped_samples": int(
                self._dropped_samples
            ),
            "queue_packets": queue_packets,
            "queue_ms_estimate": float(
                queue_ms
            ),
            "block_ms": int(
                self.block_ms
            ),
            "block_frames": int(
                self.block_frames
            ),
            "worker_pid": (
                int(
                    self.process.pid
                )
                if self.process
                is not None
                and self.process.poll()
                is None
                else None
            ),
        }
