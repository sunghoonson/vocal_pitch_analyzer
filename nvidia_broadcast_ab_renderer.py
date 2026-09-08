from __future__ import annotations

# V42_RVC_AB_ALIGNMENT_LOW_LATENCY_PATCH
# V41_RVC_AB_DEBUG_RECORD_PATCH

from pathlib import Path
import contextlib
import math
import time
import wave

import numpy as np

from nvidia_broadcast_capture import NvidiaBroadcastCapture

from rvc_ab_audio_align import align_wav_to_reference

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    SOUNDDEVICE_ERROR = f"{type(exc).__name__}: {exc}"
else:
    SOUNDDEVICE_ERROR = ""


BROADCAST_PREROLL_SECONDS = 0.35
BROADCAST_POSTROLL_SECONDS = 0.45


class NvidiaBroadcastABRenderer:
    """
    Replays two CLEAN WAV files through the selected Windows output
    (normally CABLE Input) and captures Microphone (NVIDIA Broadcast).

    NVIDIA Broadcast only exposes one microphone processing chain, so the
    plain and RVC variants cannot both be processed through Broadcast at the
    exact same instant. This renderer creates the two Broadcast variants
    sequentially after live recording stops.
    """

    def __init__(self, *, log_callback=None) -> None:
        self.log_callback = log_callback

    def _log(self, text: str) -> None:
        if self.log_callback is not None:
            self.log_callback(str(text))

    @staticmethod
    def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as reader:
            channels = int(reader.getnchannels())
            width = int(reader.getsampwidth())
            rate = int(reader.getframerate())
            frames = int(reader.getnframes())
            payload = reader.readframes(frames)

        if width != 2:
            raise RuntimeError(
                f"A/B renderer only supports PCM16 WAV: {path}"
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
            data = data[:usable].reshape(
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
                int(rate),
            ),
        )

    @staticmethod
    def _resample_linear(
        samples: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if (
            int(source_rate)
            == int(target_rate)
            or samples.size <= 1
        ):
            return np.asarray(
                samples,
                dtype=np.float32,
            )

        out_length = max(
            1,
            int(
                round(
                    samples.size
                    * float(target_rate)
                    / float(source_rate)
                )
            ),
        )

        source_x = np.linspace(
            0.0,
            1.0,
            num=samples.size,
            endpoint=False,
            dtype=np.float64,
        )
        target_x = np.linspace(
            0.0,
            1.0,
            num=out_length,
            endpoint=False,
            dtype=np.float64,
        )

        return np.interp(
            target_x,
            source_x,
            samples,
        ).astype(
            np.float32
        )

    def _play_to_output(
        self,
        *,
        wav_path: Path,
        output_device: int,
    ) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice가 없어 Broadcast A/B 재생을 할 수 없습니다. "
                + SOUNDDEVICE_ERROR
            )

        samples, source_rate = self._read_pcm16_mono(
            wav_path
        )

        info = sd.query_devices(
            int(output_device),
            "output",
        )

        max_channels = int(
            info.get(
                "max_output_channels",
                0,
            )
        )

        if max_channels <= 0:
            raise RuntimeError(
                "선택한 Windows 출력 장치가 출력 장치가 아닙니다."
            )

        target_rate = int(
            round(
                float(
                    info.get(
                        "default_samplerate",
                        source_rate,
                    )
                    or source_rate
                )
            )
        )
        target_rate = max(
            8000,
            target_rate,
        )

        if target_rate != source_rate:
            samples = self._resample_linear(
                samples,
                source_rate,
                target_rate,
            )

        channels = (
            2
            if max_channels >= 2
            else 1
        )

        stream = sd.OutputStream(
            samplerate=target_rate,
            blocksize=0,
            device=int(output_device),
            channels=channels,
            dtype="float32",
            latency="low",
        )

        stream.start()

        try:
            block = 2048
            position = 0

            while position < samples.size:
                mono = samples[
                    position:
                    position + block
                ]
                position += mono.size

                if channels == 1:
                    frame = mono.reshape(
                        -1,
                        1,
                    )
                else:
                    frame = np.repeat(
                        mono.reshape(
                            -1,
                            1,
                        ),
                        channels,
                        axis=1,
                    )

                stream.write(
                    np.asarray(
                        frame,
                        dtype=np.float32,
                    )
                )

            # Give Broadcast a short real-silence tail to flush its pipeline.
            tail_frames = int(
                target_rate
                * 0.35
            )
            if tail_frames > 0:
                stream.write(
                    np.zeros(
                        (
                            tail_frames,
                            channels,
                        ),
                        dtype=np.float32,
                    )
                )

        finally:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

    def _render_one(
        self,
        *,
        source_path: Path,
        output_device: int,
        broadcast_input_device: int,
        parent_dir: Path,
        stamp: str,
        label: str,
    ) -> Path:
        capture = NvidiaBroadcastCapture(
            log_callback=self._log
        )
        filename = (
            f"s24_broadcast_{label}_{stamp}.wav"
        )

        path = capture.start(
            device_index=int(
                broadcast_input_device
            ),
            parent_dir=parent_dir,
            stamp=stamp,
            filename=filename,
        )

        try:
            # Pre-roll captures Broadcast's steady idle state and makes sure
            # its PortAudio input stream is active before playback starts.
            time.sleep(
                BROADCAST_PREROLL_SECONDS
            )

            self._play_to_output(
                wav_path=source_path,
                output_device=int(
                    output_device
                ),
            )

            # Preserve the final Broadcast tail/latency.
            time.sleep(
                BROADCAST_POSTROLL_SECONDS
            )

        finally:
            capture.stop()

        return path

    def render_pair(
        self,
        *,
        clean_original_path: str | Path,
        clean_rvc_path: str | Path,
        output_device: int,
        broadcast_input_device: int,
        parent_dir: str | Path,
        stamp: str,
    ) -> dict[str, Path]:
        original = Path(
            clean_original_path
        ).expanduser().resolve()
        rvc = Path(
            clean_rvc_path
        ).expanduser().resolve()
        parent = Path(
            parent_dir
        ).expanduser().resolve()

        if not original.is_file():
            raise FileNotFoundError(
                original
            )

        if not rvc.is_file():
            raise FileNotFoundError(
                rvc
            )

        parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._log(
            "[RVC A/B] NVIDIA Broadcast 비교 렌더 시작. "
            "CABLE Input으로 녹음본을 순차 재생합니다."
        )

        original_out = self._render_one(
            source_path=original,
            output_device=int(
                output_device
            ),
            broadcast_input_device=int(
                broadcast_input_device
            ),
            parent_dir=parent,
            stamp=str(
                stamp
            ),
            label="original",
        )

        original_alignment = align_wav_to_reference(
            reference_path=original,
            target_path=original_out,
            max_lag_ms=1800,
            minimum_correlation=0.30,
            fade_in_ms=5.0,
        )
        self._log(
            "[RVC A/B] Broadcast ORIGINAL 자동 정렬: "
            f"lag={float(original_alignment.get('lag_ms', 0.0)):.0f}ms / "
            f"corr={float(original_alignment.get('correlation', 0.0)):.3f} / "
            f"applied={bool(original_alignment.get('applied', False))}"
        )

        rvc_out = self._render_one(
            source_path=rvc,
            output_device=int(
                output_device
            ),
            broadcast_input_device=int(
                broadcast_input_device
            ),
            parent_dir=parent,
            stamp=str(
                stamp
            ),
            label="rvc",
        )

        rvc_alignment = align_wav_to_reference(
            reference_path=rvc,
            target_path=rvc_out,
            max_lag_ms=1800,
            minimum_correlation=0.30,
            fade_in_ms=5.0,
        )
        self._log(
            "[RVC A/B] Broadcast RVC 자동 정렬: "
            f"lag={float(rvc_alignment.get('lag_ms', 0.0)):.0f}ms / "
            f"corr={float(rvc_alignment.get('correlation', 0.0)):.3f} / "
            f"applied={bool(rvc_alignment.get('applied', False))}"
        )

        self._log(
            "[RVC A/B] NVIDIA Broadcast 비교 렌더 완료."
        )

        return {
            "broadcast_original": original_out,
            "broadcast_rvc": rvc_out,
            "broadcast_original_alignment": original_alignment,
            "broadcast_rvc_alignment": rvc_alignment,
            "broadcast_preroll_ms": int(
                round(
                    BROADCAST_PREROLL_SECONDS
                    * 1000.0
                )
            ),
        }
