from __future__ import annotations

# V41_RVC_AB_DEBUG_RECORD_PATCH
# V37_NVIDIA_BROADCAST_RECORD_PATCH

from pathlib import Path
import contextlib
import math
import threading
import wave

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # runtime-specific
    sd = None
    SOUNDDEVICE_ERROR = f"{type(exc).__name__}: {exc}"
else:
    SOUNDDEVICE_ERROR = ""


def find_nvidia_broadcast_inputs() -> list[tuple[int, str]]:
    if sd is None:
        return []

    try:
        devices = sd.query_devices()
    except Exception:
        return []

    result: list[tuple[int, str]] = []

    for index, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue

        name = str(info.get("name", f"Device {index}"))
        lower = name.lower()

        if (
            "nvidia broadcast" in lower
            and "speaker" not in lower
            and "speakers" not in lower
        ):
            result.append((int(index), name))

    return result


class NvidiaBroadcastCapture:
    def __init__(self, *, log_callback=None) -> None:
        self.log_callback = log_callback
        self._lock = threading.Lock()

        self._stream = None
        self._writer: wave.Wave_write | None = None
        self._path: Path | None = None
        self._last_path: Path | None = None

        self.device_index: int | None = None
        self.device_name = ""
        self.sample_rate = 0
        self.peak_dbfs = -120.0
        self.received_frames = 0
        self.status = "idle"

    def _log(self, text: str) -> None:
        if self.log_callback is not None:
            self.log_callback(str(text))

    @staticmethod
    def _pcm16_bytes(data: np.ndarray) -> bytes:
        x = np.clip(
            np.asarray(data, dtype=np.float32).reshape(-1),
            -1.0,
            1.0,
        )
        return (x * 32767.0).astype("<i2").tobytes()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.status = str(status)

        arr = np.asarray(indata, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[1] == 1:
                mono = arr[:, 0]
            else:
                mono = np.mean(arr, axis=1, dtype=np.float32)
        else:
            mono = arr.reshape(-1)

        if mono.size <= 0:
            return

        mono = np.asarray(mono, dtype=np.float32).copy()
        np.nan_to_num(
            mono,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        peak = float(np.max(np.abs(mono)))

        self.peak_dbfs = (
            20.0 * math.log10(max(peak, 1e-8))
            if peak > 0.0
            else -120.0
        )
        self.received_frames += int(mono.size)

        with self._lock:
            writer = self._writer

            if writer is not None:
                writer.writeframesraw(
                    self._pcm16_bytes(mono)
                )

    def start(
        self,
        *,
        device_index: int,
        parent_dir: str | Path,
        stamp: str,
        filename: str | None = None,
    ) -> Path:
        self.stop()

        if sd is None:
            raise RuntimeError(
                "sounddevice가 없어 NVIDIA Broadcast 최종 마이크를 캡처할 수 없습니다. "
                + SOUNDDEVICE_ERROR
            )

        device_index = int(device_index)
        info = sd.query_devices(device_index, "input")

        if int(info.get("max_input_channels", 0)) <= 0:
            raise RuntimeError(
                "선택한 NVIDIA Broadcast 장치가 입력 장치가 아닙니다."
            )

        device_name = str(
            info.get("name", f"Device {device_index}")
        )
        sample_rate = int(
            round(
                float(
                    info.get("default_samplerate", 48000)
                    or 48000
                )
            )
        )
        sample_rate = max(8000, sample_rate)

        parent = Path(parent_dir).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        safe_filename = (
            Path(
                str(
                    filename
                )
            ).name
            if filename
            else f"s24_broadcast_{stamp}.wav"
        )
        path = parent / safe_filename

        writer = wave.open(str(path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)

        with self._lock:
            self._writer = writer
            self._path = path
            self._last_path = path

        self.device_index = device_index
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.peak_dbfs = -120.0
        self.received_frames = 0
        self.status = "starting"

        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                blocksize=0,
                device=device_index,
                channels=1,
                dtype="float32",
                latency="low",
                callback=self._callback,
            )
            stream.start()
            self._stream = stream
            self.status = "recording"

        except Exception:
            with self._lock:
                failed_writer = self._writer
                self._writer = None
                self._path = None

            if failed_writer is not None:
                with contextlib.suppress(Exception):
                    failed_writer.close()

            with contextlib.suppress(OSError):
                path.unlink()

            self.status = "open failed"
            raise

        self._log(
            "[NVIDIA Broadcast] 최종 보정 마이크 동시 녹음 시작: "
            f"{path}"
        )
        self._log(
            "[NVIDIA Broadcast] input="
            f"{device_name} / {sample_rate} Hz / mono"
        )

        return path

    def stop(self) -> Path | None:
        stream = self._stream
        self._stream = None

        # Stop PortAudio before taking the writer lock because the callback
        # itself writes under the same lock.
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

        with self._lock:
            writer = self._writer
            path = self._path
            self._writer = None
            self._path = None

            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()

        if path is not None:
            self._last_path = path
            self._log(
                "[NVIDIA Broadcast] 최종 보정 WAV 녹음 완료: "
                f"{path}"
            )

        if self.status != "idle":
            self.status = "stopped"

        return path

    def snapshot(self) -> dict:
        return {
            "recording": self._stream is not None,
            "path": str(self._path) if self._path else "",
            "last_path": str(self._last_path) if self._last_path else "",
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sample_rate": int(self.sample_rate),
            "peak_dbfs": float(self.peak_dbfs),
            "received_frames": int(self.received_frames),
            "status": str(self.status),
        }
