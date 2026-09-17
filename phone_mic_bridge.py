from __future__ import annotations

# V46_S24_VOICE_FOCUS_NATIVE_MIC_LAB_PATCH
# V45_FINAL_VOICE_MONITOR_PATCH
# V44B_NATIVE_MIC_AUTOBUILD_HOTFIX
# V44_S24_NATIVE_SCREENOFF_MIC_PATCH
# V43_REALTIME_RVC_F0_STABILITY_GUARD_PATCH
# V42_RVC_AB_ALIGNMENT_LOW_LATENCY_PATCH
# V41_RVC_AB_DEBUG_RECORD_PATCH
# V40_S24_CAMERA_VIRTUAL_WEBCAM_PATCH
# V39B_REALTIME_RVC_INDEX_HOTFIX
# V39A_PHONE_MIC_SCROLL_PATCH
# V39_REALTIME_RVC_VOICE_CHANGER_PATCH
# V38_RAW_RECORD_TOGGLE_PATCH
# V37_NVIDIA_BROADCAST_RECORD_PATCH
# V36_S24_SMART_VOICE_GAIN_PATCH
# V35_S24_NOISE_MONITOR_PATCH
# V34_S24_PHONE_MIC_BRIDGE_PATCH
#
# Galaxy S24 Ultra -> USB/ADB reverse -> Chrome localhost page
# -> raw PCM WebSocket -> Python DSP -> Windows output device.
#
# A virtual cable such as VB-CABLE can be selected as the output device
# when another Windows application needs to see the phone as a microphone.

import asyncio
import collections
import contextlib
import datetime as _dt
import html
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import wave

import numpy as np

from PySide6.QtCore import QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - depends on runtime installation
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _SOUNDDEVICE_IMPORT_ERROR = ""

from nvidia_broadcast_capture import (
    NvidiaBroadcastCapture,
    find_nvidia_broadcast_inputs,
)

from nvidia_broadcast_ab_renderer import (
    NvidiaBroadcastABRenderer,
)

from rvc_ab_audio_align import (
    align_wav_to_reference,
    write_ab_report,
)

from realtime_rvc_engine import (
    RealtimeRVCClient,
    realtime_rvc_status_text,
)

from s24_camera_bridge import (
    S24CameraController,
    camera_runtime_status_text,
)

try:
    import websockets
except Exception as exc:  # pragma: no cover - depends on runtime installation
    websockets = None
    _WEBSOCKETS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _WEBSOCKETS_IMPORT_ERROR = ""


HTTP_PORT = 8790
WS_PORT = 8791
VIDEO_WS_PORT = 8792
CAMERA_CONTROL_PORT = 8793
NATIVE_MIC_PACKAGE = "local.vocalpitch.s24mic"
NATIVE_MIC_ACTIVITY = ".MainActivity"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_PACKET_MS = 20
DEFAULT_JITTER_MS = 80


def project_root() -> Path:
    return Path(__file__).resolve().parent


def recordings_dir() -> Path:
    return project_root() / "recordings" / "phone_mic"


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def find_adb() -> Path | None:
    local_candidates = (
        project_root() / "tools" / "platform-tools" / "adb.exe",
        project_root() / "tools" / "platform-tools" / "adb",
    )

    for candidate in local_candidates:
        if candidate.is_file():
            return candidate

    found = shutil.which("adb")
    return Path(found) if found else None


def runtime_status_text() -> str:
    missing: list[str] = []

    if sd is None:
        missing.append("sounddevice")

    if websockets is None:
        missing.append("websockets")

    adb = find_adb()

    if missing:
        return (
            "Phone Mic runtime 미설치: "
            + ", ".join(missing)
            + " / SETUP_PHONE_MIC_BRIDGE.bat 실행 필요"
        )

    if adb is None:
        return (
            "Python audio runtime OK / ADB 없음 / "
            "SETUP_PHONE_MIC_BRIDGE.bat 실행 필요"
        )

    return f"Phone Mic runtime OK / ADB={adb}"


def adb_devices() -> list[tuple[str, str]]:
    adb = find_adb()

    if adb is None:
        return []

    try:
        result = subprocess.run(
            [str(adb), "devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=_creation_flags(),
        )
    except Exception:
        return []

    found: list[tuple[str, str]] = []

    for line in result.stdout.splitlines()[1:]:
        line = line.strip()

        if not line or "\t" not in line:
            continue

        serial, state = line.split("\t", 1)
        found.append((serial.strip(), state.strip()))

    return found


def _run_adb(
    args: list[str],
    *,
    serial: str | None = None,
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    adb = find_adb()

    if adb is None:
        raise RuntimeError(
            "ADB를 찾지 못했습니다. SETUP_PHONE_MIC_BRIDGE.bat을 실행하세요."
        )

    command = [str(adb)]

    if serial:
        command += ["-s", serial]

    command += list(args)

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_creation_flags(),
    )


def _first_online_android_serial() -> str:
    devices = adb_devices()

    online = [
        serial
        for serial, state in devices
        if state == "device"
    ]

    unauthorized = [
        serial
        for serial, state in devices
        if state == "unauthorized"
    ]

    if online:
        return online[
            0
        ]

    if unauthorized:
        raise RuntimeError(
            "Galaxy가 USB로 보이지만 아직 인증되지 않았습니다. "
            "휴대폰의 USB 디버깅 허용 창에서 이 PC를 허용하세요."
        )

    raise RuntimeError(
        "ADB에서 연결된 Android 기기를 찾지 못했습니다. "
        "USB 디버깅을 켜고 USB 케이블로 연결하세요."
    )


def _adb_reverse_ports(
    serial: str,
    ports: tuple[int, ...],
) -> None:
    for port in ports:
        result = _run_adb(
            [
                "reverse",
                f"tcp:{int(port)}",
                f"tcp:{int(port)}",
            ],
            serial=serial,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"adb reverse tcp:{int(port)} 실패:\n"
                + (
                    result.stderr
                    or result.stdout
                    or "(출력 없음)"
                )
            )


def native_mic_project_dir() -> Path:
    return (
        project_root()
        / "android"
        / "s24_native_mic"
    )


def native_mic_apk_candidates() -> tuple[Path, ...]:
    return (
        project_root()
        / "tools"
        / "s24_native_mic"
        / "S24NativeMic-debug.apk",
        native_mic_project_dir()
        / "app"
        / "build"
        / "outputs"
        / "apk"
        / "debug"
        / "app-debug.apk",
    )


def find_native_mic_apk() -> Path | None:
    for path in native_mic_apk_candidates():
        if path.is_file():
            return path

    return None


def native_mic_installed(
    *,
    serial: str | None = None,
) -> bool:
    try:
        serial = (
            serial
            or _first_online_android_serial()
        )
        result = _run_adb(
            [
                "shell",
                "pm",
                "path",
                NATIVE_MIC_PACKAGE,
            ],
            serial=serial,
            timeout=10.0,
        )

        return (
            result.returncode == 0
            and "package:" in result.stdout
        )

    except Exception:
        return False


def build_native_mic_apk() -> Path:
    """
    Build the Android companion automatically when the APK is missing.

    This runs on the existing Native Mic background action thread, so the
    PySide6 main loop is not blocked.
    """
    root = project_root()
    build_bat = (
        root
        / "dev_tools"
        / "BUILD_S24_NATIVE_MIC.bat"
    )

    if not build_bat.is_file():
        raise RuntimeError(
            "Native Mic APK가 없고 자동 빌드 BAT도 찾지 못했습니다.\n"
            f"필요 파일: {build_bat}"
        )

    # Use cmd.exe so the .bat can set JAVA_HOME / ANDROID_SDK_ROOT itself.
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            str(
                build_bat
            ),
        ],
        cwd=str(
            root
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900.0,
        check=False,
    )

    raw = (
        result.stdout
        or b""
    )

    # Windows console output can be UTF-8, CP949, or the active ANSI codepage.
    output = ""

    for encoding in (
        "utf-8",
        "cp949",
        "mbcs",
    ):
        try:
            output = raw.decode(
                encoding
            )
            break
        except Exception:
            continue

    if not output:
        output = raw.decode(
            "utf-8",
            errors="replace",
        )

    apk = find_native_mic_apk()

    if (
        result.returncode != 0
        or apk is None
    ):
        tail = output[
            -12000:
        ]

        raise RuntimeError(
            "S24 Native Mic APK 자동 빌드에 실패했습니다.\n\n"
            "아래는 빌드 로그 마지막 부분입니다.\n"
            "----------------------------------------\n"
            + (
                tail.strip()
                or "(출력 없음)"
            )
        )

    return apk


def install_native_mic_app(
    *,
    serial: str | None = None,
) -> tuple[str, Path]:
    serial = (
        serial
        or _first_online_android_serial()
    )

    apk = find_native_mic_apk()

    if apk is None:
        apk = build_native_mic_apk()

    result = _run_adb(
        [
            "install",
            "-r",
            str(
                apk
            ),
        ],
        serial=serial,
        timeout=120.0,
    )

    output = (
        result.stdout
        + "\n"
        + result.stderr
    ).strip()

    if (
        result.returncode != 0
        or "Success" not in output
    ):
        raise RuntimeError(
            "Native Mic APK 설치 실패:\n"
            + (
                output
                or "(출력 없음)"
            )
        )

    return (
        serial,
        apk,
    )


def launch_native_mic_app(
    *,
    auto_install: bool = True,
    auto_start: bool = True,
) -> tuple[str, bool]:
    serial = _first_online_android_serial()

    # The native app only needs the audio websocket.
    _adb_reverse_ports(
        serial,
        (
            WS_PORT,
        ),
    )

    installed_now = False

    if not native_mic_installed(
        serial=serial
    ):
        if not auto_install:
            raise RuntimeError(
                "S24 Native Mic 앱이 설치되어 있지 않습니다."
            )

        install_native_mic_app(
            serial=serial
        )
        installed_now = True

    component = (
        f"{NATIVE_MIC_PACKAGE}/"
        f"{NATIVE_MIC_ACTIVITY}"
    )

    args = [
        "shell",
        "am",
        "start",
        "-n",
        component,
    ]

    if auto_start:
        args += [
            "--ez",
            "auto_start",
            "true",
        ]

    result = _run_adb(
        args,
        serial=serial,
        timeout=20.0,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "S24 Native Mic 실행 실패:\n"
            + (
                result.stderr
                or result.stdout
                or "(출력 없음)"
            )
        )

    return (
        serial,
        installed_now,
    )


def setup_adb_reverse_and_open_browser(
    *,
    http_port: int = HTTP_PORT,
    ws_port: int = WS_PORT,
    video_ws_port: int = VIDEO_WS_PORT,
) -> tuple[str, str]:
    serial = _first_online_android_serial()

    _adb_reverse_ports(
        serial,
        (
            int(
                http_port
            ),
            int(
                ws_port
            ),
            int(
                video_ws_port
            ),
        ),
    )

    url = f"http://localhost:{int(http_port)}/"

    result = _run_adb(
        [
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            url,
        ],
        serial=serial,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "휴대폰 브라우저 열기 실패:\n"
            + (result.stderr or result.stdout or "(출력 없음)")
        )

    return serial, url


class AudioRingBuffer:
    def __init__(
        self,
        *,
        sample_rate: int,
        capacity_seconds: float = 5.0,
    ) -> None:
        self.sample_rate = max(8000, int(sample_rate))
        self.capacity = max(
            self.sample_rate,
            int(self.sample_rate * max(1.0, float(capacity_seconds))),
        )

        self._buffer = np.zeros(
            self.capacity,
            dtype=np.float32,
        )
        self._read = 0
        self._write = 0
        self._count = 0
        self._lock = threading.Lock()

        self.overflow_samples = 0
        self.underflow_samples = 0
        self.primed = False

    @property
    def available_samples(self) -> int:
        with self._lock:
            return int(self._count)

    def clear(self) -> None:
        with self._lock:
            self._read = 0
            self._write = 0
            self._count = 0
            self.primed = False

    def write(self, samples: np.ndarray) -> None:
        data = np.asarray(
            samples,
            dtype=np.float32,
        ).reshape(-1)

        if data.size <= 0:
            return

        if data.size >= self.capacity:
            data = data[-self.capacity :]

        with self._lock:
            needed = int(data.size)

            overflow = max(
                0,
                self._count + needed - self.capacity,
            )

            if overflow:
                self._read = (
                    self._read
                    + overflow
                ) % self.capacity
                self._count -= overflow
                self.overflow_samples += overflow

            first = min(
                needed,
                self.capacity - self._write,
            )

            self._buffer[
                self._write : self._write + first
            ] = data[:first]

            remaining = needed - first

            if remaining:
                self._buffer[:remaining] = data[first:]

            self._write = (
                self._write
                + needed
            ) % self.capacity
            self._count += needed

    def read(
        self,
        frames: int,
        *,
        prebuffer_samples: int,
    ) -> np.ndarray:
        frames = max(1, int(frames))
        result = np.zeros(
            frames,
            dtype=np.float32,
        )

        with self._lock:
            if (
                not self.primed
                and self._count
                < max(frames, int(prebuffer_samples))
            ):
                return result

            self.primed = True

            take = min(
                frames,
                self._count,
            )

            if take:
                first = min(
                    take,
                    self.capacity - self._read,
                )

                result[:first] = self._buffer[
                    self._read : self._read + first
                ]

                remaining = take - first

                if remaining:
                    result[
                        first : first + remaining
                    ] = self._buffer[:remaining]

                self._read = (
                    self._read
                    + take
                ) % self.capacity
                self._count -= take

            if take < frames:
                missing = frames - take
                self.underflow_samples += missing
                self.primed = False

        return result


def _resample_linear(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    source_rate = int(source_rate)
    target_rate = int(target_rate)

    data = np.asarray(
        samples,
        dtype=np.float32,
    ).reshape(-1)

    if (
        data.size <= 1
        or source_rate <= 0
        or target_rate <= 0
        or source_rate == target_rate
    ):
        return data

    target_count = max(
        1,
        int(
            round(
                data.size
                * float(target_rate)
                / float(source_rate)
            )
        ),
    )

    old_x = np.linspace(
        0.0,
        1.0,
        num=data.size,
        endpoint=False,
        dtype=np.float64,
    )
    new_x = np.linspace(
        0.0,
        1.0,
        num=target_count,
        endpoint=False,
        dtype=np.float64,
    )

    return np.interp(
        new_x,
        old_x,
        data.astype(np.float64),
    ).astype(np.float32)


class FinalVoiceMonitor:
    """
    Separate final-output monitor.

    Existing output stream:
        CLEAN/RVC -> CABLE Input -> NVIDIA Broadcast

    This monitor:
        Microphone (NVIDIA Broadcast) -> physical speakers/headphones
    """

    def __init__(
        self,
        *,
        log_callback=None,
    ) -> None:
        self.log_callback = log_callback
        self._input_stream = None
        self._output_stream = None

        self.input_device: int | None = None
        self.output_device: int | None = None
        self.input_name = ""
        self.output_name = ""

        self.input_sample_rate = DEFAULT_SAMPLE_RATE
        self.output_sample_rate = DEFAULT_SAMPLE_RATE
        self.output_channels = 2

        self.gain_db = -6.0
        self.prebuffer_ms = 30

        self.peak_dbfs = -120.0
        self.status = "idle"
        self.received_frames = 0
        self.played_frames = 0

        self.ring = AudioRingBuffer(
            sample_rate=DEFAULT_SAMPLE_RATE,
            capacity_seconds=3.0,
        )

    @property
    def running(self) -> bool:
        return (
            self._input_stream is not None
            and self._output_stream is not None
        )

    def _log(self, text: str) -> None:
        if self.log_callback is not None:
            self.log_callback(str(text))

    @staticmethod
    def _safe_rate(
        value,
        fallback: int = DEFAULT_SAMPLE_RATE,
    ) -> int:
        try:
            result = int(round(float(value)))
        except Exception:
            result = int(fallback)

        return max(8000, min(result, 192000))

    def _input_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            self.status = "input: " + str(status)

        arr = np.asarray(indata, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[1] == 1:
                mono = arr[:, 0]
            else:
                mono = np.mean(
                    arr,
                    axis=1,
                    dtype=np.float32,
                )
        else:
            mono = arr.reshape(-1)

        if mono.size <= 0:
            return

        mono = np.asarray(
            mono,
            dtype=np.float32,
        ).copy()

        np.nan_to_num(
            mono,
            copy=False,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        peak = float(
            np.max(
                np.abs(mono)
            )
        )
        self.peak_dbfs = (
            20.0 * math.log10(max(peak, 1e-8))
            if peak > 0.0
            else -120.0
        )

        if self.input_sample_rate != self.output_sample_rate:
            mono = _resample_linear(
                mono,
                self.input_sample_rate,
                self.output_sample_rate,
            )

        gain = float(
            10.0
            ** (
                float(self.gain_db)
                / 20.0
            )
        )

        if abs(gain - 1.0) > 1e-5:
            mono *= gain

        np.clip(
            mono,
            -1.0,
            1.0,
            out=mono,
        )

        self.ring.write(mono)
        self.received_frames += int(frames)

    def _output_callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            self.status = "output: " + str(status)

        prebuffer = max(
            int(frames),
            int(
                self.output_sample_rate
                * float(self.prebuffer_ms)
                / 1000.0
            ),
        )

        mono = self.ring.read(
            int(frames),
            prebuffer_samples=prebuffer,
        )

        if outdata.ndim == 1:
            outdata[:] = mono
        else:
            for channel in range(
                outdata.shape[1]
            ):
                outdata[:, channel] = mono

        self.played_frames += int(frames)

    def start(
        self,
        *,
        input_device: int,
        output_device: int,
        gain_db: float = -6.0,
        prebuffer_ms: int = 30,
    ) -> None:
        self.stop()

        if sd is None:
            raise RuntimeError(
                "sounddevice가 없어 최종 보이스 모니터를 시작할 수 없습니다. "
                + (_SOUNDDEVICE_IMPORT_ERROR or "unknown")
            )

        input_device = int(input_device)
        output_device = int(output_device)

        input_info = sd.query_devices(
            input_device,
            "input",
        )
        output_info = sd.query_devices(
            output_device,
            "output",
        )

        if int(input_info.get("max_input_channels", 0)) <= 0:
            raise RuntimeError(
                "선택한 NVIDIA Broadcast 최종 마이크가 입력 장치가 아닙니다."
            )

        max_output_channels = int(
            output_info.get(
                "max_output_channels",
                0,
            )
        )
        if max_output_channels <= 0:
            raise RuntimeError(
                "선택한 최종 모니터 장치가 출력 장치가 아닙니다."
            )

        input_default = self._safe_rate(
            input_info.get(
                "default_samplerate",
                DEFAULT_SAMPLE_RATE,
            )
        )
        output_default = self._safe_rate(
            output_info.get(
                "default_samplerate",
                DEFAULT_SAMPLE_RATE,
            )
        )
        channels = 2 if max_output_channels >= 2 else 1

        common_rate = None
        rate_candidates = []

        for rate in (
            DEFAULT_SAMPLE_RATE,
            input_default,
            output_default,
        ):
            rate = int(rate)
            if rate not in rate_candidates:
                rate_candidates.append(rate)

        for rate in rate_candidates:
            try:
                sd.check_input_settings(
                    device=input_device,
                    channels=1,
                    dtype="float32",
                    samplerate=rate,
                )
                sd.check_output_settings(
                    device=output_device,
                    channels=channels,
                    dtype="float32",
                    samplerate=rate,
                )
                common_rate = int(rate)
                break
            except Exception:
                continue

        if common_rate is None:
            input_rate = int(input_default)
            output_rate = int(output_default)
        else:
            input_rate = common_rate
            output_rate = common_rate

        self.input_device = input_device
        self.output_device = output_device
        self.input_name = str(
            input_info.get(
                "name",
                f"Device {input_device}",
            )
        )
        self.output_name = str(
            output_info.get(
                "name",
                f"Device {output_device}",
            )
        )
        self.input_sample_rate = int(input_rate)
        self.output_sample_rate = int(output_rate)
        self.output_channels = int(channels)
        self.gain_db = max(
            -40.0,
            min(float(gain_db), 12.0),
        )
        self.prebuffer_ms = max(
            10,
            min(int(prebuffer_ms), 250),
        )

        self.ring = AudioRingBuffer(
            sample_rate=self.output_sample_rate,
            capacity_seconds=3.0,
        )
        self.peak_dbfs = -120.0
        self.received_frames = 0
        self.played_frames = 0
        self.status = "starting"

        input_stream = None
        output_stream = None

        try:
            output_stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                blocksize=0,
                device=output_device,
                channels=channels,
                dtype="float32",
                latency="low",
                callback=self._output_callback,
            )
            input_stream = sd.InputStream(
                samplerate=self.input_sample_rate,
                blocksize=0,
                device=input_device,
                channels=1,
                dtype="float32",
                latency="low",
                callback=self._input_callback,
            )

            output_stream.start()
            input_stream.start()

            self._output_stream = output_stream
            self._input_stream = input_stream
            self.status = "monitoring"

        except Exception:
            if input_stream is not None:
                with contextlib.suppress(Exception):
                    input_stream.stop()
                with contextlib.suppress(Exception):
                    input_stream.close()

            if output_stream is not None:
                with contextlib.suppress(Exception):
                    output_stream.stop()
                with contextlib.suppress(Exception):
                    output_stream.close()

            self._input_stream = None
            self._output_stream = None
            self.status = "open failed"
            self.ring.clear()
            raise

        rate_text = (
            f"{self.input_sample_rate}→{self.output_sample_rate} Hz"
            if self.input_sample_rate != self.output_sample_rate
            else f"{self.output_sample_rate} Hz"
        )

        self._log(
            "[Final Voice Monitor] ON: "
            f"{self.input_name} → {self.output_name} / "
            f"{rate_text} / gain={self.gain_db:+.1f}dB / "
            f"prebuffer={self.prebuffer_ms}ms"
        )

    def set_gain_db(
        self,
        value: float,
    ) -> None:
        self.gain_db = max(
            -40.0,
            min(float(value), 12.0),
        )

    def stop(self) -> None:
        input_stream = self._input_stream
        output_stream = self._output_stream
        self._input_stream = None
        self._output_stream = None

        if input_stream is not None:
            with contextlib.suppress(Exception):
                input_stream.stop()
            with contextlib.suppress(Exception):
                input_stream.close()

        if output_stream is not None:
            with contextlib.suppress(Exception):
                output_stream.stop()
            with contextlib.suppress(Exception):
                output_stream.close()

        self.ring.clear()

        if self.status != "idle":
            self.status = "stopped"

    def snapshot(self) -> dict:
        rate = max(
            1,
            int(self.output_sample_rate),
        )

        return {
            "running": bool(self.running),
            "status": str(self.status),
            "input_device": self.input_device,
            "output_device": self.output_device,
            "input_name": str(self.input_name),
            "output_name": str(self.output_name),
            "input_sample_rate": int(
                self.input_sample_rate
            ),
            "output_sample_rate": int(
                self.output_sample_rate
            ),
            "gain_db": float(self.gain_db),
            "prebuffer_ms": int(
                self.prebuffer_ms
            ),
            "peak_dbfs": float(self.peak_dbfs),
            "received_frames": int(
                self.received_frames
            ),
            "played_frames": int(
                self.played_frames
            ),
            "buffer_ms": (
                self.ring.available_samples
                * 1000.0
                / float(rate)
            ),
            "underflow_ms": (
                self.ring.underflow_samples
                * 1000.0
                / float(rate)
            ),
        }


class SimpleDSP:
    """
    Intentionally conservative real-time DSP.

    Raw WAV recording is written BEFORE these filters, so training data
    can stay close to the phone capture.
    """

    def __init__(
        self,
        sample_rate: int,
    ) -> None:
        self.sample_rate = max(8000, int(sample_rate))
        self.gain_db = 0.0
        self.highpass_enabled = False
        self.highpass_hz = 70.0
        self.gate_enabled = False
        self.gate_threshold_db = -55.0

        # Conservative suppressor for short broadband transients such as
        # mouse clicks / keyboard impacts during low-level or silent gaps.
        # It intentionally avoids strong speech frames.
        self.transient_suppression_enabled = True
        self.transient_reduction_db = 10.0
        self.transient_hits = 0

        # v3.6 Smart Voice Gain:
        # slow upward gain / fast downward correction, followed by a
        # packet-safe limiter. This is designed for desk-distance speech.
        self.smart_gain_enabled = True
        self.smart_gain_target_db = -20.0
        self.smart_gain_max_boost_db = 18.0
        self.smart_gain_max_cut_db = 12.0
        self.smart_gain_speech_floor_db = -48.0
        self.limiter_enabled = True
        self.limiter_ceiling_db = -1.0
        self.smart_gain_current_db = 0.0
        self.smart_gain_last_input_rms_db = -120.0
        self.smart_gain_last_output_rms_db = -120.0
        self.limiter_reduction_db = 0.0

        self._hp_x1 = 0.0
        self._hp_y1 = 0.0
        self._gate_gain = 1.0

    def reset(self) -> None:
        self._hp_x1 = 0.0
        self._hp_y1 = 0.0
        self._gate_gain = 1.0
        self.transient_hits = 0
        self.smart_gain_current_db = 0.0
        self.smart_gain_last_input_rms_db = -120.0
        self.smart_gain_last_output_rms_db = -120.0
        self.limiter_reduction_db = 0.0

    def process(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(
            data,
            dtype=np.float32,
        ).reshape(-1).copy()

        if x.size <= 0:
            return x

        if self.highpass_enabled:
            cutoff = max(
                20.0,
                min(
                    float(self.highpass_hz),
                    self.sample_rate * 0.40,
                ),
            )
            dt = 1.0 / float(self.sample_rate)
            rc = 1.0 / (
                2.0
                * math.pi
                * cutoff
            )
            alpha = rc / (rc + dt)

            x1 = float(self._hp_x1)
            y1 = float(self._hp_y1)

            out = np.empty_like(x)

            # First-order high-pass. The loop is small (usually 20ms packet).
            for i, sample in enumerate(x):
                current = float(sample)
                y = alpha * (
                    y1
                    + current
                    - x1
                )
                out[i] = y
                x1 = current
                y1 = y

            self._hp_x1 = x1
            self._hp_y1 = y1
            x = out

        if self.gate_enabled:
            rms = float(
                np.sqrt(
                    np.mean(
                        x.astype(np.float64) ** 2
                    )
                    + 1e-12
                )
            )
            db = 20.0 * math.log10(
                max(rms, 1e-8)
            )

            target = (
                1.0
                if db >= float(self.gate_threshold_db)
                else 0.10
            )

            # Soft expander rather than a hard gate so consonants don't
            # disappear abruptly.
            smoothing = (
                0.35
                if target > self._gate_gain
                else 0.08
            )
            self._gate_gain += (
                target - self._gate_gain
            ) * smoothing
            x *= float(self._gate_gain)

        if (
            self.transient_suppression_enabled
            and x.size >= 128
        ):
            rms = float(
                np.sqrt(
                    np.mean(
                        x.astype(np.float64) ** 2
                    )
                    + 1e-12
                )
            )
            peak = float(
                np.max(
                    np.abs(x)
                )
                + 1e-12
            )
            rms_db = (
                20.0
                * math.log10(
                    max(
                        rms,
                        1e-8,
                    )
                )
            )
            crest_db = (
                20.0
                * math.log10(
                    max(
                        peak / max(rms, 1e-8),
                        1.0,
                    )
                )
            )

            # Only inspect quieter frames. This protects normal vowels,
            # plosives and emphasized consonants from unnecessary attenuation.
            if (
                rms_db < -34.0
                and crest_db > 10.5
            ):
                window = np.hanning(
                    int(x.size)
                ).astype(
                    np.float32
                )
                spectrum = np.abs(
                    np.fft.rfft(
                        x * window
                    )
                )
                frequencies = np.fft.rfftfreq(
                    int(x.size),
                    d=1.0 / float(self.sample_rate),
                )
                total = float(
                    np.sum(
                        spectrum
                    )
                    + 1e-12
                )
                centroid_hz = float(
                    np.sum(
                        spectrum
                        * frequencies
                    )
                    / total
                )

                if centroid_hz >= 3200.0:
                    transient_gain = (
                        10.0
                        ** (
                            -abs(
                                float(
                                    self.transient_reduction_db
                                )
                            )
                            / 20.0
                        )
                    )
                    x *= float(
                        transient_gain
                    )
                    self.transient_hits += 1

        gain = 10.0 ** (
            float(self.gain_db)
            / 20.0
        )
        x *= float(gain)

        # ----------------------------------------------------
        # v3.6 Smart Voice Gain
        # ----------------------------------------------------
        #
        # The phone can be 40~80 cm away on a desk. Fixed gain alone
        # either stays too quiet or clips when the user moves closer.
        # This leveler follows speech RMS toward a target, increases
        # gain slowly, reduces it quickly, and does NOT chase silence.
        rms = float(
            np.sqrt(
                np.mean(
                    x.astype(np.float64) ** 2
                )
                + 1e-12
            )
        )
        input_rms_db = (
            20.0
            * math.log10(
                max(
                    rms,
                    1e-8,
                )
            )
        )
        self.smart_gain_last_input_rms_db = float(
            input_rms_db
        )

        if self.smart_gain_enabled:
            if input_rms_db >= float(
                self.smart_gain_speech_floor_db
            ):
                desired_db = float(
                    self.smart_gain_target_db
                ) - float(
                    input_rms_db
                )
                desired_db = max(
                    -abs(
                        float(
                            self.smart_gain_max_cut_db
                        )
                    ),
                    min(
                        abs(
                            float(
                                self.smart_gain_max_boost_db
                            )
                        ),
                        desired_db,
                    ),
                )

                # When the signal suddenly gets louder, reduce gain fast
                # to avoid clipping. Raise gain slowly to avoid pumping.
                if desired_db < self.smart_gain_current_db:
                    alpha = 0.35
                else:
                    alpha = 0.055

                self.smart_gain_current_db += (
                    desired_db
                    - self.smart_gain_current_db
                ) * alpha

            # If the frame is below the speech floor, hold the current
            # gain instead of turning room noise up toward the target.
            auto_gain = 10.0 ** (
                float(
                    self.smart_gain_current_db
                )
                / 20.0
            )
            x *= float(
                auto_gain
            )
        else:
            self.smart_gain_current_db = 0.0

        # Packet-safe peak limiter. The whole 20 ms packet is scaled
        # together, which avoids hard sample clipping.
        self.limiter_reduction_db = 0.0

        if self.limiter_enabled:
            ceiling = 10.0 ** (
                float(
                    self.limiter_ceiling_db
                )
                / 20.0
            )
            peak = float(
                np.max(
                    np.abs(
                        x
                    )
                )
                + 1e-12
            )

            if peak > ceiling:
                scale = float(
                    ceiling
                    / peak
                )
                x *= scale
                self.limiter_reduction_db = float(
                    20.0
                    * math.log10(
                        max(
                            scale,
                            1e-8,
                        )
                    )
                )

        output_rms = float(
            np.sqrt(
                np.mean(
                    x.astype(np.float64) ** 2
                )
                + 1e-12
            )
        )
        self.smart_gain_last_output_rms_db = float(
            20.0
            * math.log10(
                max(
                    output_rms,
                    1e-8,
                )
            )
        )

        # Final numerical safety.
        np.clip(
            x,
            -0.999,
            0.999,
            out=x,
        )

        return x


def _phone_page_html(
    *,
    ws_port: int,
    video_ws_port: int = VIDEO_WS_PORT,
) -> str:
    template = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>S24 AV Bridge</title>
<style>
:root { color-scheme: dark; }
body { margin:0; padding:20px; font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif; background:#111318; color:#f2f3f5; }
.card { max-width:820px; margin:0 auto 14px; padding:18px; border-radius:16px; background:#1d2128; }
h1 { font-size:24px; margin:0 0 12px; } h2 { font-size:17px; margin:0 0 10px; }
p { line-height:1.55; color:#cfd3da; }
button,select { width:100%; min-height:50px; margin-top:10px; border:0; border-radius:12px; font-size:16px; padding:10px 12px; }
button { font-weight:700; }.primary{background:#4d8dff;color:white}.secondary{background:#3a3e46;color:white}
#meter{width:100%;height:18px;background:#30343b;border-radius:10px;overflow:hidden}#bar{width:0%;height:100%;background:#6ea1ff;transition:width 80ms linear}
.small{font-size:13px;color:#9fa7b2;word-break:break-all}.ok{color:#76d39b}.warn{color:#ffcc66}.bad{color:#ff7f7f}
.option-row{display:flex;align-items:center;gap:10px;margin-top:10px;color:#d8dce3}.option-row input{width:22px;height:22px}
#videoPreview{width:100%;max-height:60vh;margin-top:12px;border-radius:12px;background:#050608;object-fit:contain}
</style>
</head>
<body>
<div class="card"><h1>Galaxy S24 Ultra → PC AV Bridge</h1><p>USB/ADB reverse로 마이크 PCM과 카메라 JPEG 프레임을 PC로 전송합니다. 오디오와 카메라는 독립적으로 시작/중지할 수 있습니다.</p><div id="secure" class="small"></div></div>
<div class="card">
<h2>마이크</h2><select id="device"></select>
<label class="option-row"><input id="noiseSuppression" type="checkbox" checked><span>브라우저 Noise Suppression</span></label>
<label class="option-row"><input id="echoCancellation" type="checkbox"><span>Echo Cancellation</span></label>
<label class="option-row"><input id="autoGainControl" type="checkbox"><span>Auto Gain Control</span></label>
<button id="start" class="primary">마이크 시작</button><button id="stop" class="secondary" disabled>마이크 중지</button><p id="state" class="small">대기 중</p><div id="meter"><div id="bar"></div></div>
</div>
<div class="card">
<h2>카메라</h2><p class="small">아래 버튼을 한 번 눌러 카메라 권한과 PC 영상 브리지를 연결하세요. 이후 카메라 선택, 해상도, FPS, 미러링, 보정, 가상배경은 PC 앱에서 제어합니다.</p>
<button id="cameraConnect" class="primary">카메라 권한 허용 / PC 제어 연결</button><button id="cameraStop" class="secondary" disabled>카메라 중지</button><p id="cameraState" class="small">카메라 대기 중</p><video id="videoPreview" autoplay playsinline muted></video>
</div>
<div class="card"><h2>권장</h2><p class="small">카메라 사용 중에는 화면을 켜 두는 것이 가장 안정적입니다. 영상 처리는 PC에서 수행합니다.</p></div>
<script>
const AUDIO_WS_PORT=__AUDIO_WS_PORT__, VIDEO_WS_PORT=__VIDEO_WS_PORT__, PACKET_FRAMES=960;
let ws=null,stream=null,context=null,source=null,worklet=null,sink=null,wakeLock=null,meterTimer=null,analyser=null;
let videoWs=null,videoStream=null,videoTimer=null,videoEncoding=false;
let videoCanvas=document.createElement("canvas"),videoContext=videoCanvas.getContext("2d",{alpha:false});
let lastCameraConfig={deviceId:"",width:1280,height:720,fps:15,quality:.72};
const deviceSelect=document.getElementById("device"),startButton=document.getElementById("start"),stopButton=document.getElementById("stop"),state=document.getElementById("state"),bar=document.getElementById("bar"),secure=document.getElementById("secure"),noiseSuppressionCheck=document.getElementById("noiseSuppression"),echoCancellationCheck=document.getElementById("echoCancellation"),autoGainControlCheck=document.getElementById("autoGainControl");
const cameraConnectButton=document.getElementById("cameraConnect"),cameraStopButton=document.getElementById("cameraStop"),cameraState=document.getElementById("cameraState"),videoPreview=document.getElementById("videoPreview");
secure.textContent="Secure context: "+window.isSecureContext+" / page="+location.href; secure.className=window.isSecureContext?"small ok":"small bad";
function setState(t,c="small"){state.textContent=t;state.className=c}
function setCameraState(t,c="small"){cameraState.textContent=t;cameraState.className=c;if(videoWs&&videoWs.readyState===WebSocket.OPEN){try{videoWs.send(JSON.stringify({type:"camera_state",state:t,running:!!videoStream}))}catch(_){}}}
async function refreshDevices(){let ds=[];try{ds=await navigator.mediaDevices.enumerateDevices()}catch(e){setState("장치 목록 실패: "+e,"small warn");return}const inputs=ds.filter(d=>d.kind==="audioinput"),prev=deviceSelect.value;deviceSelect.innerHTML="";inputs.forEach((d,i)=>{const o=document.createElement("option");o.value=d.deviceId;o.textContent=d.label||("마이크 "+(i+1));deviceSelect.appendChild(o)});if([...deviceSelect.options].some(o=>o.value===prev))deviceSelect.value=prev}
async function sendCameraDevices(){if(!navigator.mediaDevices)return;let ds=[];try{ds=await navigator.mediaDevices.enumerateDevices()}catch(_){return}const cams=ds.filter(d=>d.kind==="videoinput").map((d,i)=>({deviceId:d.deviceId,label:d.label||("카메라 "+(i+1))}));if(videoWs&&videoWs.readyState===WebSocket.OPEN)videoWs.send(JSON.stringify({type:"camera_devices",devices:cams}))}
function connectSocket(){return new Promise((resolve,reject)=>{const socket=new WebSocket("ws://localhost:"+AUDIO_WS_PORT+"/");socket.binaryType="arraybuffer";const to=setTimeout(()=>{try{socket.close()}catch(_){}reject(new Error("Audio WebSocket 연결 시간 초과"))},5000);socket.onopen=()=>{clearTimeout(to);ws=socket;resolve(socket)};socket.onerror=()=>{clearTimeout(to);reject(new Error("PC Audio WebSocket 연결 실패"))};socket.onclose=()=>{if(ws===socket)ws=null}})}
function connectVideoSocket(){return new Promise((resolve,reject)=>{if(videoWs&&videoWs.readyState===WebSocket.OPEN){resolve(videoWs);return}const socket=new WebSocket("ws://localhost:"+VIDEO_WS_PORT+"/");socket.binaryType="arraybuffer";const to=setTimeout(()=>{try{socket.close()}catch(_){}reject(new Error("Video WebSocket 연결 시간 초과"))},5000);socket.onopen=async()=>{clearTimeout(to);videoWs=socket;setCameraState("PC 영상 브리지 연결됨","small ok");await sendCameraDevices();resolve(socket)};socket.onerror=()=>{clearTimeout(to);reject(new Error("PC 영상 브리지 연결 실패. PC 앱에서 카메라 워커를 먼저 시작하세요."))};socket.onclose=()=>{if(videoWs===socket)videoWs=null;setCameraState("PC 영상 브리지 연결 끊김","small warn")};socket.onmessage=async e=>{if(typeof e.data!=="string")return;let p;try{p=JSON.parse(e.data)}catch(_){return}if(p.type!=="camera_control")return;if(p.action==="stop"){await stopCamera();return}if(p.action==="start"){try{await startCamera({deviceId:p.deviceId||"",width:Number(p.width||1280),height:Number(p.height||720),fps:Number(p.fps||15),quality:Number(p.quality||.72)})}catch(err){setCameraState("카메라 시작 실패: "+err,"small bad")}}if(p.action==="enumerate")await sendCameraDevices()}})}
async function requestWakeLock(){try{if("wakeLock" in navigator)wakeLock=await navigator.wakeLock.request("screen")}catch(_){}}
async function startMic(){if(!window.isSecureContext||!navigator.mediaDevices)throw new Error("브라우저가 마이크 API를 허용하지 않습니다.");if(!ws||ws.readyState!==WebSocket.OPEN)await connectSocket();const selected=deviceSelect.value;stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:selected?{exact:selected}:undefined,channelCount:{ideal:1},sampleRate:{ideal:48000},echoCancellation:echoCancellationCheck.checked,noiseSuppression:noiseSuppressionCheck.checked,autoGainControl:autoGainControlCheck.checked},video:false});await refreshDevices();context=new AudioContext({sampleRate:48000,latencyHint:"interactive"});const code=`class PhoneMicProcessor extends AudioWorkletProcessor{constructor(){super();this.packet=new Float32Array(${PACKET_FRAMES});this.offset=0}process(inputs){const input=inputs[0];if(!input||input.length===0||!input[0])return true;const channel=input[0];let pos=0;while(pos<channel.length){const take=Math.min(channel.length-pos,this.packet.length-this.offset);this.packet.set(channel.subarray(pos,pos+take),this.offset);this.offset+=take;pos+=take;if(this.offset>=this.packet.length){const ready=this.packet;this.packet=new Float32Array(${PACKET_FRAMES});this.offset=0;this.port.postMessage(ready.buffer,[ready.buffer])}}return true}}registerProcessor("phone-mic-processor",PhoneMicProcessor);`;const blob=new Blob([code],{type:"application/javascript"}),url=URL.createObjectURL(blob);await context.audioWorklet.addModule(url);URL.revokeObjectURL(url);source=context.createMediaStreamSource(stream);worklet=new AudioWorkletNode(context,"phone-mic-processor",{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});analyser=context.createAnalyser();analyser.fftSize=1024;sink=context.createGain();sink.gain.value=0;source.connect(analyser);source.connect(worklet);worklet.connect(sink);sink.connect(context.destination);worklet.port.onmessage=e=>{if(ws&&ws.readyState===WebSocket.OPEN)ws.send(e.data)};const track=stream.getAudioTracks()[0],settings=track?track.getSettings():{};ws.send(JSON.stringify({type:"hello",sampleRate:context.sampleRate,packetFrames:PACKET_FRAMES,deviceLabel:track?track.label:"",trackSettings:settings,requestedNoiseSuppression:noiseSuppressionCheck.checked,requestedEchoCancellation:echoCancellationCheck.checked,requestedAutoGainControl:autoGainControlCheck.checked}));const meterData=new Float32Array(analyser.fftSize);meterTimer=setInterval(()=>{if(!analyser)return;analyser.getFloatTimeDomainData(meterData);let sum=0;for(let i=0;i<meterData.length;i++)sum+=meterData[i]*meterData[i];const rms=Math.sqrt(sum/meterData.length),db=20*Math.log10(Math.max(rms,1e-6)),pct=Math.max(0,Math.min(100,(db+60)/60*100));bar.style.width=pct+"%"},80);await context.resume();await requestWakeLock();startButton.disabled=true;stopButton.disabled=false;noiseSuppressionCheck.disabled=true;echoCancellationCheck.disabled=true;autoGainControlCheck.disabled=true;setState("전송 중 / "+(track?track.label:"microphone")+" / "+context.sampleRate+" Hz","small ok")}
async function stopMic(){if(meterTimer){clearInterval(meterTimer);meterTimer=null}if(worklet){try{worklet.disconnect()}catch(_){}worklet=null}if(source){try{source.disconnect()}catch(_){}source=null}if(sink){try{sink.disconnect()}catch(_){}sink=null}analyser=null;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}if(context){try{await context.close()}catch(_){}context=null}if(ws){try{ws.close()}catch(_){}ws=null}startButton.disabled=false;stopButton.disabled=true;noiseSuppressionCheck.disabled=false;echoCancellationCheck.disabled=false;autoGainControlCheck.disabled=false;bar.style.width="0%";setState("중지됨")}
async function startCamera(config){if(!window.isSecureContext||!navigator.mediaDevices)throw new Error("브라우저가 카메라 API를 허용하지 않습니다.");if(!videoWs||videoWs.readyState!==WebSocket.OPEN)await connectVideoSocket();await stopCamera(false);lastCameraConfig={deviceId:String(config.deviceId||""),width:Math.max(320,Number(config.width||1280)),height:Math.max(240,Number(config.height||720)),fps:Math.max(5,Number(config.fps||15)),quality:Math.max(.30,Math.min(.95,Number(config.quality||.72)))};const vc={width:{ideal:lastCameraConfig.width},height:{ideal:lastCameraConfig.height},frameRate:{ideal:lastCameraConfig.fps,max:lastCameraConfig.fps}};if(lastCameraConfig.deviceId)vc.deviceId={exact:lastCameraConfig.deviceId};else vc.facingMode={ideal:"environment"};videoStream=await navigator.mediaDevices.getUserMedia({audio:false,video:vc});videoPreview.srcObject=videoStream;await videoPreview.play();const track=videoStream.getVideoTracks()[0],settings=track?track.getSettings():{},aw=Number(settings.width||lastCameraConfig.width),ah=Number(settings.height||lastCameraConfig.height),af=Number(settings.frameRate||lastCameraConfig.fps);videoCanvas.width=aw;videoCanvas.height=ah;await sendCameraDevices();videoWs.send(JSON.stringify({type:"camera_hello",label:track?track.label:"",settings}));const interval=Math.max(16,Math.round(1000/Math.max(5,lastCameraConfig.fps)));videoTimer=setInterval(async()=>{if(!videoStream||!videoWs||videoWs.readyState!==WebSocket.OPEN||videoEncoding)return;videoEncoding=true;try{videoContext.drawImage(videoPreview,0,0,videoCanvas.width,videoCanvas.height);const blob=await new Promise(resolve=>videoCanvas.toBlob(resolve,"image/jpeg",lastCameraConfig.quality));if(blob&&videoWs&&videoWs.readyState===WebSocket.OPEN)videoWs.send(await blob.arrayBuffer())}finally{videoEncoding=false}},interval);cameraConnectButton.textContent="카메라 브리지 연결됨 / PC에서 제어";cameraStopButton.disabled=false;setCameraState("전송 중 / "+(track?track.label:"camera")+" / "+aw+"x"+ah+" / "+af.toFixed(1)+"fps","small ok");await requestWakeLock()}
async function stopCamera(notify=true){if(videoTimer){clearInterval(videoTimer);videoTimer=null}videoEncoding=false;if(videoStream){videoStream.getTracks().forEach(t=>t.stop());videoStream=null}videoPreview.srcObject=null;cameraStopButton.disabled=true;if(notify)setCameraState("카메라 중지됨")}
cameraConnectButton.addEventListener("click",async()=>{try{await connectVideoSocket();await startCamera(lastCameraConfig)}catch(err){setCameraState("카메라 연결 실패: "+err,"small bad")}});cameraStopButton.addEventListener("click",async()=>await stopCamera());startButton.addEventListener("click",async()=>{try{setState("마이크 권한/오디오 장치 준비 중...");await startMic()}catch(err){setState("시작 실패: "+err,"small bad");try{await stopMic()}catch(_){}}});stopButton.addEventListener("click",async()=>await stopMic());navigator.mediaDevices.addEventListener("devicechange",async()=>{await refreshDevices();await sendCameraDevices()});document.addEventListener("visibilitychange",async()=>{if(document.visibilityState==="visible"&&(!wakeLock||wakeLock.released))await requestWakeLock()});window.addEventListener("beforeunload",()=>{if(ws)try{ws.close()}catch(_){}if(videoWs)try{videoWs.close()}catch(_){}});
(async()=>{if(!navigator.mediaDevices){setState("mediaDevices API 없음","small bad");setCameraState("mediaDevices API 없음","small bad");return}await refreshDevices();try{await connectVideoSocket()}catch(_){setCameraState("PC 카메라 워커 대기 중 / PC에서 카메라 기능을 시작한 뒤 이 버튼을 누르세요.","small warn")}})();
</script>
</body>
</html>
'''
    return template.replace("__AUDIO_WS_PORT__", str(int(ws_port))).replace("__VIDEO_WS_PORT__", str(int(video_ws_port)))


class _PhonePageHandler(BaseHTTPRequestHandler):
    page_html = ""

    def do_GET(self) -> None:
        body = self.page_html.encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(body)),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.send_header(
            "Permissions-Policy",
            "microphone=(self), camera=(self)",
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        return


class PhoneMicRuntime:
    def __init__(
        self,
        *,
        http_port: int = HTTP_PORT,
        ws_port: int = WS_PORT,
    ) -> None:
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)

        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.output_device: int | None = None
        # v3.5: monitor is OFF by default. Recording/bridge can run silently.
        self.output_enabled = False
        self.jitter_ms = DEFAULT_JITTER_MS
        # v3.8: RAW is optional and defaults OFF.
        self.record_raw_copy = False
        self.record_clean_copy = True
        self.broadcast_record_enabled = True
        self.broadcast_input_device: int | None = None

        # v4.5: final NVIDIA Broadcast voice -> local speaker/headphone monitor.
        self.final_monitor_enabled = False
        self.final_monitor_output_device: int | None = None
        self.final_monitor_gain_db = -6.0
        self.final_monitor_prebuffer_ms = 30
        self.final_voice_monitor = FinalVoiceMonitor(
            log_callback=self.log
        )

        # v4.1 RVC A/B debug recorder.
        self.rvc_ab_debug_enabled = False
        self._ab_clean_original_wave: wave.Wave_write | None = None
        self._ab_clean_rvc_wave: wave.Wave_write | None = None
        self._ab_clean_original_path: Path | None = None
        self._ab_clean_rvc_path: Path | None = None
        self._ab_last_clean_original_path: Path | None = None
        self._ab_last_clean_rvc_path: Path | None = None
        self._ab_last_broadcast_original_path: Path | None = None
        self._ab_last_broadcast_rvc_path: Path | None = None
        self._ab_last_report_path: Path | None = None
        self._ab_clean_rvc_alignment: dict = {}
        self._ab_render_state = "idle"
        self._ab_render_error = ""
        self._ab_rendering = False
        self._ab_render_thread: threading.Thread | None = None
        self._ab_renderer = NvidiaBroadcastABRenderer(
            log_callback=self.log
        )

        self.ring = AudioRingBuffer(
            sample_rate=self.sample_rate,
            capacity_seconds=5.0,
        )
        self.dsp = SimpleDSP(
            sample_rate=self.sample_rate,
        )

        self._settings_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._record_lock = threading.Lock()

        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_stop_async: asyncio.Event | None = None
        self._ws_server = None
        self._audio_stream = None

        self._running = False
        self._log_queue: queue.Queue[str] = queue.Queue()

        self._record_wave: wave.Wave_write | None = None
        self._record_path: Path | None = None
        self._last_raw_record_path: Path | None = None
        self._last_clean_record_path: Path | None = None
        self._record_session_stamp = ""
        self._clean_record_wave: wave.Wave_write | None = None
        self._clean_record_path: Path | None = None

        self.broadcast_capture = NvidiaBroadcastCapture(
            log_callback=self.log
        )

        self.realtime_rvc_enabled = False
        self.realtime_rvc_client: RealtimeRVCClient | None = None
        self.realtime_rvc_state = "stopped"
        self.realtime_rvc_error = ""
        self.realtime_rvc_model = ""
        self.realtime_rvc_index = ""
        self.realtime_rvc_pitch = 0
        self.realtime_rvc_index_rate = 0.35
        self.realtime_rvc_block_ms = 200
        self.realtime_rvc_f0_guard_enabled = True
        self.realtime_rvc_f0_diagnostic_enabled = True
        self.realtime_rvc_rmvpe_threshold = 0.05
        self.realtime_rvc_f0_quiet_rms_db = -48.0
        self.realtime_rvc_f0_min_voiced_ratio = 0.15
        self.realtime_rvc_f0_max_gap_ms = 30
        self.realtime_rvc_f0_min_run_ms = 40
        self._realtime_rvc_load_thread: threading.Thread | None = None

        self.client_connected = False
        self.client_device_label = ""
        self.client_sample_rate = 0
        self.packet_frames = 0
        self.peak_dbfs = -120.0
        self.received_samples = 0
        self.received_packets = 0
        self.last_packet_monotonic = 0.0

    @property
    def running(self) -> bool:
        return bool(self._running)

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._log_queue.put(
            f"[{stamp}] {str(message).strip()}"
        )

    def drain_logs(self) -> list[str]:
        result: list[str] = []

        while True:
            try:
                result.append(
                    self._log_queue.get_nowait()
                )
            except queue.Empty:
                break

        return result

    def configure(
        self,
        *,
        output_device: int | None,
        output_enabled: bool,
        jitter_ms: int,
        gain_db: float,
        highpass_enabled: bool,
        highpass_hz: float,
        gate_enabled: bool,
        gate_threshold_db: float,
        transient_suppression_enabled: bool,
        transient_reduction_db: float,
        smart_gain_enabled: bool,
        smart_gain_target_db: float,
        smart_gain_max_boost_db: float,
        limiter_enabled: bool,
        limiter_ceiling_db: float,
        record_raw_copy: bool,
        record_clean_copy: bool,
        broadcast_record_enabled: bool,
        broadcast_input_device: int | None,
        final_monitor_enabled: bool,
        final_monitor_output_device: int | None,
        final_monitor_gain_db: float,
        realtime_rvc_enabled: bool,
        rvc_ab_debug_enabled: bool,
    ) -> None:
        with self._settings_lock:
            self.output_device = (
                int(output_device)
                if output_device is not None
                else None
            )
            self.output_enabled = bool(output_enabled)
            self.jitter_ms = max(
                20,
                min(
                    int(jitter_ms),
                    500,
                ),
            )
            self.dsp.gain_db = float(gain_db)
            self.dsp.highpass_enabled = bool(highpass_enabled)
            self.dsp.highpass_hz = float(highpass_hz)
            self.dsp.gate_enabled = bool(gate_enabled)
            self.dsp.gate_threshold_db = float(gate_threshold_db)
            self.dsp.transient_suppression_enabled = bool(
                transient_suppression_enabled
            )
            self.dsp.transient_reduction_db = float(
                transient_reduction_db
            )
            self.dsp.smart_gain_enabled = bool(
                smart_gain_enabled
            )
            self.dsp.smart_gain_target_db = float(
                smart_gain_target_db
            )
            self.dsp.smart_gain_max_boost_db = float(
                smart_gain_max_boost_db
            )
            self.dsp.limiter_enabled = bool(
                limiter_enabled
            )
            self.dsp.limiter_ceiling_db = float(
                limiter_ceiling_db
            )
            self.record_raw_copy = bool(
                record_raw_copy
            )
            self.record_clean_copy = bool(
                record_clean_copy
            )
            self.broadcast_record_enabled = bool(
                broadcast_record_enabled
            )
            self.broadcast_input_device = (
                int(broadcast_input_device)
                if broadcast_input_device is not None
                else None
            )
            self.final_monitor_enabled = bool(
                final_monitor_enabled
            )
            self.final_monitor_output_device = (
                int(final_monitor_output_device)
                if final_monitor_output_device is not None
                else None
            )
            self.final_monitor_gain_db = max(
                -40.0,
                min(
                    float(final_monitor_gain_db),
                    12.0,
                ),
            )
            self.final_voice_monitor.set_gain_db(
                self.final_monitor_gain_db
            )
            self.realtime_rvc_enabled = bool(
                realtime_rvc_enabled
            )
            self.rvc_ab_debug_enabled = bool(
                rvc_ab_debug_enabled
            )

    def _output_callback(
        self,
        outdata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            self.log(
                f"Windows output status: {status}"
            )

        with self._settings_lock:
            jitter_ms = int(self.jitter_ms)
            monitor_enabled = bool(
                self.output_enabled
            )

        if not monitor_enabled:
            outdata.fill(
                0.0
            )
            return

        prebuffer = int(
            self.sample_rate
            * jitter_ms
            / 1000.0
        )

        mono = self.ring.read(
            int(frames),
            prebuffer_samples=prebuffer,
        )

        if outdata.ndim == 1:
            outdata[:] = mono
            return

        for channel in range(
            outdata.shape[1]
        ):
            outdata[:, channel] = mono

    def _open_output_stream(self) -> None:
        if self._audio_stream is not None:
            return

        if not self.output_enabled:
            self.log(
                "Windows 출력 비활성 / 수신·녹음 전용으로 시작"
            )
            return

        if sd is None:
            raise RuntimeError(
                "sounddevice가 설치되지 않았습니다: "
                + (_SOUNDDEVICE_IMPORT_ERROR or "unknown")
            )

        device = self.output_device

        info = sd.query_devices(
            device,
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
                "선택한 장치는 출력 장치가 아닙니다."
            )

        default_sr = float(
            info.get(
                "default_samplerate",
                DEFAULT_SAMPLE_RATE,
            )
            or DEFAULT_SAMPLE_RATE
        )

        # 48 kHz is preferred, but use the device's native default if
        # attempting 48 kHz would be unreasonable.
        target_sr = (
            DEFAULT_SAMPLE_RATE
            if 32000 <= default_sr <= 192000
            else int(round(default_sr))
        )

        if abs(default_sr - DEFAULT_SAMPLE_RATE) > 100.0:
            target_sr = int(
                round(default_sr)
            )

        self.sample_rate = max(
            8000,
            int(target_sr),
        )

        self.ring = AudioRingBuffer(
            sample_rate=self.sample_rate,
            capacity_seconds=5.0,
        )
        self.dsp.sample_rate = self.sample_rate
        self.dsp.reset()

        channels = 2 if max_channels >= 2 else 1

        self._audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=0,
            device=device,
            channels=channels,
            dtype="float32",
            latency="low",
            callback=self._output_callback,
        )
        self._audio_stream.start()

        self.log(
            "Windows output 시작: "
            f"{info.get('name', device)} / "
            f"{self.sample_rate} Hz / {channels}ch"
        )

    def _close_output_stream(self) -> None:
        stream = self._audio_stream
        self._audio_stream = None

        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

        self.ring.clear()

    def set_monitor_enabled(
        self,
        enabled: bool,
    ) -> None:
        enabled = bool(
            enabled
        )

        with self._settings_lock:
            self.output_enabled = enabled

        if not self._running:
            return

        if enabled:
            self._open_output_stream()
            self.log(
                "실시간 모니터 출력 ON"
            )
        else:
            self._close_output_stream()
            self.log(
                "실시간 모니터 출력 OFF / 녹음은 계속됩니다."
            )

    def _open_final_voice_monitor(
        self,
    ) -> None:
        if not self.final_monitor_enabled:
            return

        if self.final_voice_monitor.running:
            return

        input_device = self.broadcast_input_device
        output_device = self.final_monitor_output_device

        if input_device is None:
            raise RuntimeError(
                "NVIDIA Broadcast 최종 마이크 입력을 선택하세요."
            )

        if output_device is None:
            raise RuntimeError(
                "최종 보이스를 들을 스피커/헤드폰을 선택하세요."
            )

        if (
            self.output_enabled
            and self.output_device is not None
            and int(output_device) == int(self.output_device)
        ):
            self.log(
                "[Final Voice Monitor] 경고: 최종 모니터 출력이 "
                "CLEAN/RVC 전달 출력과 같습니다. CABLE Input을 선택했다면 "
                "루프가 생길 수 있습니다."
            )

        self.final_voice_monitor.start(
            input_device=int(input_device),
            output_device=int(output_device),
            gain_db=float(
                self.final_monitor_gain_db
            ),
            prebuffer_ms=int(
                self.final_monitor_prebuffer_ms
            ),
        )

    def _close_final_voice_monitor(
        self,
    ) -> None:
        was_running = bool(
            self.final_voice_monitor.running
        )
        self.final_voice_monitor.stop()

        if was_running:
            self.log(
                "[Final Voice Monitor] OFF"
            )

    def set_final_monitor_enabled(
        self,
        enabled: bool,
    ) -> None:
        enabled = bool(enabled)

        with self._settings_lock:
            self.final_monitor_enabled = enabled

        if not self._running:
            return

        if enabled:
            self._open_final_voice_monitor()
        else:
            self._close_final_voice_monitor()

    def restart_final_voice_monitor(
        self,
    ) -> None:
        if not self._running:
            return

        with self._settings_lock:
            enabled = bool(
                self.final_monitor_enabled
            )

        self._close_final_voice_monitor()

        if enabled:
            self._open_final_voice_monitor()

    def set_final_monitor_gain_db(
        self,
        gain_db: float,
    ) -> None:
        value = max(
            -40.0,
            min(float(gain_db), 12.0),
        )

        with self._settings_lock:
            self.final_monitor_gain_db = value

        self.final_voice_monitor.set_gain_db(
            value
        )

    def _on_realtime_rvc_audio(
        self,
        data: np.ndarray,
    ) -> None:
        audio = np.asarray(
            data,
            dtype=np.float32,
        ).reshape(
            -1
        )

        with self._record_lock:
            ab_writer = self._ab_clean_rvc_wave

            if ab_writer is not None:
                ab_writer.writeframesraw(
                    self._float_to_pcm16_bytes(
                        audio
                    )
                )

        with self._settings_lock:
            enabled = bool(
                self.realtime_rvc_enabled
            )
            output_enabled = bool(
                self.output_enabled
            )

        if (
            enabled
            and output_enabled
            and not self._ab_rendering
        ):
            self.ring.write(
                audio
            )

    def start_realtime_rvc_async(
        self,
        *,
        model_path: str | Path,
        index_path: str | Path | None,
        pitch: int,
        index_rate: float,
        block_ms: int,
        f0_guard_enabled: bool,
        f0_diagnostic_enabled: bool,
        rmvpe_threshold: float,
        f0_quiet_rms_db: float,
        f0_min_voiced_ratio: float,
        f0_max_gap_ms: int,
        f0_min_run_ms: int,
    ) -> None:
        model = str(
            model_path
            or ""
        ).strip()

        if not model:
            raise RuntimeError(
                "Realtime RVC .pth 모델을 선택하세요."
            )

        if not self.running:
            raise RuntimeError(
                "PC 브리지를 먼저 시작하세요. "
                "출력 장치 sample rate가 결정된 뒤 Realtime RVC를 로드합니다."
            )

        if self.realtime_rvc_state == "loading":
            return

        self.stop_realtime_rvc()

        self.realtime_rvc_state = "loading"
        self.realtime_rvc_error = ""
        self.realtime_rvc_model = model
        self.realtime_rvc_index = str(
            index_path
            or ""
        )
        self.realtime_rvc_pitch = int(
            pitch
        )
        self.realtime_rvc_index_rate = float(
            index_rate
        )
        self.realtime_rvc_block_ms = int(
            block_ms
        )
        self.realtime_rvc_f0_guard_enabled = bool(
            f0_guard_enabled
        )
        self.realtime_rvc_f0_diagnostic_enabled = bool(
            f0_diagnostic_enabled
        )
        self.realtime_rvc_rmvpe_threshold = float(
            rmvpe_threshold
        )
        self.realtime_rvc_f0_quiet_rms_db = float(
            f0_quiet_rms_db
        )
        self.realtime_rvc_f0_min_voiced_ratio = float(
            f0_min_voiced_ratio
        )
        self.realtime_rvc_f0_max_gap_ms = int(
            f0_max_gap_ms
        )
        self.realtime_rvc_f0_min_run_ms = int(
            f0_min_run_ms
        )

        self.log(
            "[Realtime RVC] 모델 로딩 시작. "
            "HuBERT/RMVPE/RVC를 GPU에 유지합니다."
        )

        def loader() -> None:
            client = RealtimeRVCClient(
                log_callback=self.log,
                audio_callback=self._on_realtime_rvc_audio,
            )

            try:
                client.start(
                    model_path=model,
                    index_path=(
                        str(
                            index_path
                        )
                        if index_path
                        else None
                    ),
                    sample_rate=int(
                        self.sample_rate
                    ),
                    pitch=int(
                        pitch
                    ),
                    index_rate=float(
                        index_rate
                    ),
                    block_ms=int(
                        block_ms
                    ),
                    crossfade_ms=40,
                    extra_ms=1000,
                    f0_method="rmvpe",
                    f0_guard_enabled=bool(
                        f0_guard_enabled
                    ),
                    f0_diagnostic_enabled=bool(
                        f0_diagnostic_enabled
                    ),
                    rmvpe_threshold=float(
                        rmvpe_threshold
                    ),
                    f0_quiet_rms_db=float(
                        f0_quiet_rms_db
                    ),
                    f0_min_voiced_ratio=float(
                        f0_min_voiced_ratio
                    ),
                    f0_max_gap_ms=int(
                        f0_max_gap_ms
                    ),
                    f0_min_run_ms=int(
                        f0_min_run_ms
                    ),
                    timeout=120.0,
                )

                self.realtime_rvc_client = client
                self.realtime_rvc_state = "ready"
                self.realtime_rvc_error = ""
                self.ring.clear()

                self.log(
                    "[Realtime RVC] READY - "
                    "이제 CLEAN 신호가 RVC를 통과한 뒤 Windows 출력으로 전달됩니다."
                )

            except Exception as exc:
                client.stop()
                self.realtime_rvc_client = None
                self.realtime_rvc_state = "error"
                self.realtime_rvc_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.log(
                    "[Realtime RVC] 로드 실패 - CLEAN bypass 유지: "
                    + self.realtime_rvc_error
                )

        self._realtime_rvc_load_thread = threading.Thread(
            target=loader,
            name="RealtimeRVCLoad",
            daemon=True,
        )
        self._realtime_rvc_load_thread.start()

    def stop_realtime_rvc(
        self,
    ) -> None:
        client = self.realtime_rvc_client
        self.realtime_rvc_client = None

        if client is not None:
            try:
                client.stop()
            except Exception:
                pass

        if self.realtime_rvc_state != "stopped":
            self.log(
                "[Realtime RVC] 엔진 중지"
            )

        self.realtime_rvc_state = "stopped"
        self.realtime_rvc_error = ""
        self.ring.clear()

    def _realtime_rvc_snapshot(
        self,
    ) -> dict:
        client = self.realtime_rvc_client

        if client is None:
            return {
                "state": str(
                    self.realtime_rvc_state
                ),
                "enabled": bool(
                    self.realtime_rvc_enabled
                ),
                "error": str(
                    self.realtime_rvc_error
                ),
                "ready": False,
                "last_roundtrip_ms": 0.0,
                "processed_blocks": 0,
                "dropped_samples": 0,
                "queue_packets": 0,
                "queue_ms_estimate": 0.0,
                "block_ms": int(
                    self.realtime_rvc_block_ms
                ),
                "worker_pid": None,
            }

        snap = client.snapshot()
        snap.update(
            {
                "state": str(
                    self.realtime_rvc_state
                ),
                "enabled": bool(
                    self.realtime_rvc_enabled
                ),
            }
        )
        return snap

    def _start_http(self) -> None:
        page = _phone_page_html(
            ws_port=self.ws_port,
            video_ws_port=VIDEO_WS_PORT,
        )

        handler_type = type(
            "PhonePageHandler",
            (_PhonePageHandler,),
            {"page_html": page},
        )

        self._http_server = ThreadingHTTPServer(
            ("127.0.0.1", self.http_port),
            handler_type,
        )

        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="PhoneMicHTTP",
            daemon=True,
        )
        self._http_thread.start()

        self.log(
            f"Phone page server: http://127.0.0.1:{self.http_port}/"
        )

    async def _ws_handler(
        self,
        websocket,
        path=None,
    ) -> None:
        with self._stats_lock:
            self.client_connected = True
            self.client_device_label = ""
            self.client_sample_rate = 0
            self.packet_frames = 0

        self.ring.clear()
        self.dsp.reset()

        remote = getattr(
            websocket,
            "remote_address",
            None,
        )
        self.log(
            f"Galaxy WebSocket 연결: {remote}"
        )

        try:
            async for message in websocket:
                if isinstance(
                    message,
                    str,
                ):
                    try:
                        payload = json.loads(
                            message
                        )
                    except json.JSONDecodeError:
                        continue

                    if payload.get("type") == "hello":
                        rate = int(
                            payload.get(
                                "sampleRate",
                                DEFAULT_SAMPLE_RATE,
                            )
                            or DEFAULT_SAMPLE_RATE
                        )
                        packet_frames = int(
                            payload.get(
                                "packetFrames",
                                0,
                            )
                            or 0
                        )
                        label = str(
                            payload.get(
                                "deviceLabel",
                                "",
                            )
                            or ""
                        )
                        requested_ns = bool(
                            payload.get(
                                "requestedNoiseSuppression",
                                False,
                            )
                        )
                        requested_aec = bool(
                            payload.get(
                                "requestedEchoCancellation",
                                False,
                            )
                        )
                        requested_agc = bool(
                            payload.get(
                                "requestedAutoGainControl",
                                False,
                            )
                        )

                        source_profile = str(
                            payload.get(
                                "audioSourceName",
                                "",
                            )
                            or ""
                        )
                        direction_name = str(
                            payload.get(
                                "microphoneDirectionName",
                                "",
                            )
                            or ""
                        )
                        field_zoom = float(
                            payload.get(
                                "microphoneFieldZoom",
                                0.0,
                            )
                            or 0.0
                        )
                        direction_applied = bool(
                            payload.get(
                                "directionApplied",
                                False,
                            )
                        )
                        field_applied = bool(
                            payload.get(
                                "fieldApplied",
                                False,
                            )
                        )
                        actual_ns = bool(
                            payload.get(
                                "actualNoiseSuppression",
                                requested_ns,
                            )
                        )
                        actual_aec = bool(
                            payload.get(
                                "actualEchoCancellation",
                                requested_aec,
                            )
                        )
                        actual_agc = bool(
                            payload.get(
                                "actualAutoGainControl",
                                requested_agc,
                            )
                        )
                        active_mics = str(
                            payload.get(
                                "activeMicrophones",
                                "",
                            )
                            or ""
                        )

                        with self._stats_lock:
                            self.client_sample_rate = rate
                            self.packet_frames = packet_frames
                            self.client_device_label = label

                        self.log(
                            "Galaxy audio: "
                            f"{label or 'microphone'} / "
                            f"{rate} Hz / "
                            f"{packet_frames} frames / "
                            f"source={source_profile or 'unknown'} / "
                            f"direction={direction_name or 'default'}"
                            f"({'OK' if direction_applied else 'HAL?'}) / "
                            f"field={field_zoom:+.2f}"
                            f"({'OK' if field_applied else 'HAL?'}) / "
                            f"NS={'ON' if actual_ns else 'OFF'} / "
                            f"AEC={'ON' if actual_aec else 'OFF'} / "
                            f"AGC={'ON' if actual_agc else 'OFF'}"
                        )

                        if active_mics:
                            self.log(
                                "Galaxy active microphones: "
                                + active_mics
                            )

                    continue

                if not isinstance(
                    message,
                    (bytes, bytearray, memoryview),
                ):
                    continue

                raw = np.frombuffer(
                    message,
                    dtype="<f4",
                )

                if raw.size <= 0:
                    continue

                data = np.asarray(
                    raw,
                    dtype=np.float32,
                ).copy()

                np.nan_to_num(
                    data,
                    copy=False,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                with self._stats_lock:
                    source_rate = (
                        int(self.client_sample_rate)
                        if self.client_sample_rate > 0
                        else DEFAULT_SAMPLE_RATE
                    )

                if source_rate != self.sample_rate:
                    data = _resample_linear(
                        data,
                        source_rate,
                        self.sample_rate,
                    )

                peak = float(
                    np.max(
                        np.abs(data)
                    )
                )
                if peak > 0.0:
                    db = 20.0 * math.log10(
                        max(peak, 1e-8)
                    )
                else:
                    db = -120.0

                with self._settings_lock:
                    processed = self.dsp.process(
                        data
                    )
                    output_enabled = bool(
                        self.output_enabled
                    )

                # Always preserve the capture stream as RAW.
                # Optionally save a second CLEAN file with the desktop DSP.
                self._write_recording(
                    data,
                    processed=processed,
                )

                rvc_client = (
                    self.realtime_rvc_client
                )
                rvc_active = (
                    bool(
                        self.realtime_rvc_enabled
                    )
                    and self.realtime_rvc_state
                    == "ready"
                    and rvc_client
                    is not None
                    and rvc_client.ready
                )

                with self._record_lock:
                    ab_recording = (
                        self._ab_clean_rvc_wave
                        is not None
                    )

                if not self._ab_rendering:
                    if (
                        rvc_active
                        and (
                            output_enabled
                            or ab_recording
                        )
                    ):
                        rvc_client.push(
                            processed
                        )
                    elif output_enabled:
                        # Safe bypass while RVC is disabled/loading/failed.
                        self.ring.write(
                            processed
                        )

                with self._stats_lock:
                    self.peak_dbfs = float(db)
                    self.received_samples += int(
                        data.size
                    )
                    self.received_packets += 1
                    self.last_packet_monotonic = time.monotonic()

        except Exception as exc:
            self.log(
                "Galaxy WebSocket 종료: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            with self._stats_lock:
                self.client_connected = False

            self.ring.clear()

    async def _ws_main(self) -> None:
        assert websockets is not None

        self._ws_stop_async = asyncio.Event()

        self._ws_server = await websockets.serve(
            self._ws_handler,
            "127.0.0.1",
            self.ws_port,
            max_size=4 * 1024 * 1024,
            ping_interval=10,
            ping_timeout=20,
        )

        self.log(
            f"PCM WebSocket server: ws://127.0.0.1:{self.ws_port}/"
        )

        await self._ws_stop_async.wait()

        self._ws_server.close()
        await self._ws_server.wait_closed()

    def _ws_thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._ws_loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self._ws_main()
            )
        except Exception as exc:
            self.log(
                "WebSocket server 오류: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(
                    loop.shutdown_asyncgens()
                )

            loop.close()
            self._ws_loop = None
            self._ws_stop_async = None

    def start(self) -> None:
        if self._running:
            return

        if websockets is None:
            raise RuntimeError(
                "websockets가 설치되지 않았습니다: "
                + (_WEBSOCKETS_IMPORT_ERROR or "unknown")
            )

        # Output stream first so a device error is reported before servers
        # begin accepting microphone data.
        self._open_output_stream()

        if self.final_monitor_enabled:
            try:
                self._open_final_voice_monitor()
            except Exception as exc:
                self.log(
                    "[Final Voice Monitor] 시작 실패: "
                    f"{type(exc).__name__}: {exc}"
                )

        self._start_http()

        self._ws_thread = threading.Thread(
            target=self._ws_thread_main,
            name="PhoneMicWebSocket",
            daemon=True,
        )
        self._ws_thread.start()

        deadline = time.monotonic() + 3.0

        while (
            self._ws_loop is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        self._running = True
        self.log(
            "Phone Mic Bridge 준비 완료"
        )

    def stop(self) -> None:
        if not self._running:
            return

        self.stop_recording()
        self.stop_realtime_rvc()
        self._close_final_voice_monitor()

        loop = self._ws_loop
        stop_event = self._ws_stop_async

        if (
            loop is not None
            and stop_event is not None
        ):
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(
                    stop_event.set
                )

        if self._http_server is not None:
            with contextlib.suppress(Exception):
                self._http_server.shutdown()
            with contextlib.suppress(Exception):
                self._http_server.server_close()
            self._http_server = None

        self._close_output_stream()

        if (
            self._ws_thread is not None
            and self._ws_thread.is_alive()
        ):
            self._ws_thread.join(
                timeout=3.0
            )

        if (
            self._http_thread is not None
            and self._http_thread.is_alive()
        ):
            self._http_thread.join(
                timeout=2.0
            )

        self._ws_thread = None
        self._http_thread = None

        self.ring.clear()

        with self._stats_lock:
            self.client_connected = False

        self._running = False
        self.log(
            "Phone Mic Bridge 중지"
        )

    def start_recording(
        self,
        path: Path | None = None,
    ) -> Path:
        with self._record_lock:
            if (
                self._record_wave is not None
                or self._clean_record_wave is not None
            ):
                active = (
                    self._record_path
                    or self._clean_record_path
                )
                if active is not None:
                    return active

            recordings_dir().mkdir(
                parents=True,
                exist_ok=True,
            )

            if path is None:
                stamp = _dt.datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                parent = recordings_dir()
            else:
                requested = Path(
                    path
                ).expanduser().resolve()
                parent = requested.parent
                stem = requested.stem

                if stem.startswith(
                    "s24_raw_"
                ):
                    stamp = stem[
                        len(
                            "s24_raw_"
                        ):
                    ]
                else:
                    stamp = _dt.datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

            parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            raw_path = (
                parent
                / f"s24_raw_{stamp}.wav"
            )
            clean_path = (
                parent
                / f"s24_clean_{stamp}.wav"
            )
            ab_clean_original_path = (
                parent
                / f"s24_clean_original_{stamp}.wav"
            )
            ab_clean_rvc_path = (
                parent
                / f"s24_clean_rvc_{stamp}.wav"
            )

            self._record_session_stamp = stamp
            self._last_raw_record_path = None
            self._last_clean_record_path = None
            self._ab_last_clean_original_path = None
            self._ab_last_clean_rvc_path = None
            self._ab_last_broadcast_original_path = None
            self._ab_last_broadcast_rvc_path = None
            self._ab_last_report_path = None
            self._ab_clean_rvc_alignment = {}
            self._ab_render_state = "idle"
            self._ab_render_error = ""

            ab_enabled = bool(
                self.rvc_ab_debug_enabled
            )

            if ab_enabled:
                rvc_client = self.realtime_rvc_client

                if not (
                    self.realtime_rvc_enabled
                    and self.realtime_rvc_state
                    == "ready"
                    and rvc_client is not None
                    and rvc_client.ready
                ):
                    raise RuntimeError(
                        "RVC A/B 디버그 녹음은 Realtime RVC가 READY 상태여야 합니다."
                    )

                if self.output_device is None:
                    raise RuntimeError(
                        "RVC A/B Broadcast 렌더용 Windows 출력 장치를 선택하세요."
                    )

                if self.broadcast_input_device is None:
                    raise RuntimeError(
                        "RVC A/B Broadcast 렌더용 NVIDIA Broadcast 최종 마이크 입력을 선택하세요."
                    )

            if self.record_raw_copy:
                raw_writer = wave.open(
                    str(
                        raw_path
                    ),
                    "wb",
                )
                raw_writer.setnchannels(
                    1
                )
                raw_writer.setsampwidth(
                    2
                )
                raw_writer.setframerate(
                    int(
                        self.sample_rate
                    )
                )
                self._record_wave = raw_writer
                self._record_path = raw_path
                self._last_raw_record_path = raw_path
            else:
                self._record_wave = None
                self._record_path = None

            if (
                self.record_clean_copy
                and not ab_enabled
            ):
                clean_writer = wave.open(
                    str(
                        clean_path
                    ),
                    "wb",
                )
                clean_writer.setnchannels(
                    1
                )
                clean_writer.setsampwidth(
                    2
                )
                clean_writer.setframerate(
                    int(
                        self.sample_rate
                    )
                )
                self._clean_record_wave = clean_writer
                self._clean_record_path = clean_path
                self._last_clean_record_path = clean_path
            else:
                self._clean_record_wave = None
                self._clean_record_path = None

            if ab_enabled:
                original_writer = wave.open(
                    str(
                        ab_clean_original_path
                    ),
                    "wb",
                )
                original_writer.setnchannels(
                    1
                )
                original_writer.setsampwidth(
                    2
                )
                original_writer.setframerate(
                    int(
                        self.sample_rate
                    )
                )

                rvc_writer = wave.open(
                    str(
                        ab_clean_rvc_path
                    ),
                    "wb",
                )
                rvc_writer.setnchannels(
                    1
                )
                rvc_writer.setsampwidth(
                    2
                )
                rvc_writer.setframerate(
                    int(
                        self.sample_rate
                    )
                )

                self._ab_clean_original_wave = original_writer
                self._ab_clean_rvc_wave = rvc_writer
                self._ab_clean_original_path = ab_clean_original_path
                self._ab_clean_rvc_path = ab_clean_rvc_path
                self._ab_last_clean_original_path = ab_clean_original_path
                self._ab_last_clean_rvc_path = ab_clean_rvc_path
                self._ab_render_state = "recording"
            else:
                self._ab_clean_original_wave = None
                self._ab_clean_rvc_wave = None
                self._ab_clean_original_path = None
                self._ab_clean_rvc_path = None

        if self._record_path is not None:
            self.log(
                f"RAW WAV 녹음 시작: {self._record_path}"
            )
        else:
            self.log(
                "RAW WAV 저장 OFF"
            )

        if self._clean_record_path is not None:
            self.log(
                "CLEAN WAV 동시 녹음 시작: "
                f"{self._clean_record_path}"
            )
        else:
            self.log(
                "CLEAN WAV 저장 OFF"
            )

        if self.rvc_ab_debug_enabled:
            self.log(
                "[RVC A/B] CLEAN 원본/RVC 동시 녹음 시작: "
                f"{self._ab_clean_original_path} / "
                f"{self._ab_clean_rvc_path}"
            )
            self.log(
                "[RVC A/B] Broadcast 원본/RVC는 녹음 종료 후 "
                "NVIDIA Broadcast를 순차 통과시켜 자동 생성합니다."
            )

        if (
            self.broadcast_record_enabled
            and not self.rvc_ab_debug_enabled
        ):
            if self.broadcast_input_device is None:
                self.log(
                    "[NVIDIA Broadcast] 입력 장치가 선택되지 않아 "
                    "Broadcast WAV를 생략합니다."
                )
            else:
                try:
                    self.broadcast_capture.start(
                        device_index=self.broadcast_input_device,
                        parent_dir=parent,
                        stamp=stamp,
                    )
                except Exception as exc:
                    self.log(
                        "[NVIDIA Broadcast] 최종 보정 마이크 캡처 시작 실패: "
                        f"{type(exc).__name__}: {exc}"
                    )

        primary = (
            self._record_path
            or self._clean_record_path
            or self._ab_clean_original_path
            or Path(
                self.broadcast_capture.snapshot().get(
                    "path",
                    "",
                )
            )
        )

        if not primary or str(primary) == ".":
            raise RuntimeError(
                "저장할 녹음 형식이 없습니다. RAW/CLEAN/BROADCAST 또는 RVC A/B 중 하나 이상을 켜세요."
            )

        return Path(
            primary
        )

    @staticmethod
    def _float_to_pcm16_bytes(
        data: np.ndarray,
    ) -> bytes:
        pcm = np.clip(
            np.asarray(
                data,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )
        pcm16 = (
            pcm
            * 32767.0
        ).astype(
            "<i2"
        )
        return pcm16.tobytes()

    def _write_recording(
        self,
        raw_data: np.ndarray,
        *,
        processed: np.ndarray | None = None,
    ) -> None:
        with self._record_lock:
            raw_writer = self._record_wave
            clean_writer = self._clean_record_wave
            ab_original_writer = self._ab_clean_original_wave

            if raw_writer is not None:
                raw_writer.writeframesraw(
                    self._float_to_pcm16_bytes(
                        raw_data
                    )
                )

            if (
                clean_writer is not None
                and processed is not None
            ):
                clean_writer.writeframesraw(
                    self._float_to_pcm16_bytes(
                        processed
                    )
                )

            if (
                ab_original_writer is not None
                and processed is not None
            ):
                ab_original_writer.writeframesraw(
                    self._float_to_pcm16_bytes(
                        processed
                    )
                )

    def _start_ab_broadcast_render(
        self,
        *,
        clean_original_path: Path,
        clean_rvc_path: Path,
        parent_dir: Path,
        stamp: str,
    ) -> None:
        output_device = self.output_device
        broadcast_input_device = self.broadcast_input_device

        if (
            output_device is None
            or broadcast_input_device is None
        ):
            self._ab_render_state = "error"
            self._ab_render_error = (
                "A/B Broadcast 렌더 장치가 선택되지 않았습니다."
            )
            return

        if (
            self._ab_render_thread is not None
            and self._ab_render_thread.is_alive()
        ):
            self.log(
                "[RVC A/B] 이전 Broadcast 렌더가 아직 진행 중입니다."
            )
            return

        def worker() -> None:
            self._ab_render_state = "rendering_broadcast"
            self._ab_render_error = ""
            self._ab_rendering = True

            with self._settings_lock:
                reopen_output = bool(
                    self.output_enabled
                )
                reopen_final_monitor = bool(
                    self.final_monitor_enabled
                    and self.final_voice_monitor.running
                )

            self.log(
                "[RVC A/B] Broadcast 2종 렌더 중에는 "
                "실시간 CABLE 출력이 잠시 중지됩니다."
            )
            self.log(
                "[RVC A/B] Discord/게임이 Broadcast 마이크를 사용 중이면 "
                "테스트 녹음 음성이 다시 송신될 수 있으니 테스트 중에는 송신을 꺼두세요."
            )

            try:
                if reopen_final_monitor:
                    self._close_final_voice_monitor()

                self._close_output_stream()
                self.ring.clear()

                outputs = self._ab_renderer.render_pair(
                    clean_original_path=clean_original_path,
                    clean_rvc_path=clean_rvc_path,
                    output_device=int(
                        output_device
                    ),
                    broadcast_input_device=int(
                        broadcast_input_device
                    ),
                    parent_dir=parent_dir,
                    stamp=stamp,
                )

                self._ab_last_broadcast_original_path = Path(
                    outputs[
                        "broadcast_original"
                    ]
                )
                self._ab_last_broadcast_rvc_path = Path(
                    outputs[
                        "broadcast_rvc"
                    ]
                )

                report_payload = {
                    "version": "v4.2",
                    "stamp": str(
                        stamp
                    ),
                    "clean_original": str(
                        clean_original_path
                    ),
                    "clean_rvc": str(
                        clean_rvc_path
                    ),
                    "clean_rvc_alignment": dict(
                        self._ab_clean_rvc_alignment
                    ),
                    "broadcast_original": str(
                        self._ab_last_broadcast_original_path
                    ),
                    "broadcast_rvc": str(
                        self._ab_last_broadcast_rvc_path
                    ),
                    "broadcast_original_alignment": outputs.get(
                        "broadcast_original_alignment",
                        {},
                    ),
                    "broadcast_rvc_alignment": outputs.get(
                        "broadcast_rvc_alignment",
                        {},
                    ),
                    "broadcast_preroll_ms": outputs.get(
                        "broadcast_preroll_ms",
                        0,
                    ),
                }

                report_path = (
                    Path(
                        parent_dir
                    )
                    / f"s24_rvc_ab_report_{stamp}.json"
                )
                self._ab_last_report_path = write_ab_report(
                    path=report_path,
                    payload=report_payload,
                )
                self._ab_render_state = "done"

                self.log(
                    "[RVC A/B] 4종 비교 파일 준비 완료:"
                )
                self.log(
                    f"  CLEAN ORIGINAL: {clean_original_path}"
                )
                self.log(
                    f"  CLEAN RVC: {clean_rvc_path}"
                )
                self.log(
                    "  BROADCAST ORIGINAL: "
                    f"{self._ab_last_broadcast_original_path}"
                )
                self.log(
                    "  BROADCAST RVC: "
                    f"{self._ab_last_broadcast_rvc_path}"
                )
                self.log(
                    "  REPORT: "
                    f"{self._ab_last_report_path}"
                )

            except Exception as exc:
                self._ab_render_state = "error"
                self._ab_render_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.log(
                    "[RVC A/B] Broadcast 비교 렌더 실패: "
                    + self._ab_render_error
                )

            finally:
                self._ab_rendering = False
                self.ring.clear()

                if (
                    reopen_output
                    and self._running
                ):
                    try:
                        self._open_output_stream()
                    except Exception as exc:
                        self.log(
                            "[RVC A/B] 실시간 출력 복구 실패: "
                            f"{type(exc).__name__}: {exc}"
                        )

                if (
                    reopen_final_monitor
                    and self._running
                ):
                    try:
                        self._open_final_voice_monitor()
                    except Exception as exc:
                        self.log(
                            "[RVC A/B] 최종 보이스 모니터 복구 실패: "
                            f"{type(exc).__name__}: {exc}"
                        )

        self._ab_render_thread = threading.Thread(
            target=worker,
            name="RVCBroadcastABRender",
            daemon=True,
        )
        self._ab_render_thread.start()

    def stop_recording(self) -> Path | None:
        broadcast_path = self.broadcast_capture.stop()

        with self._record_lock:
            raw_writer = self._record_wave
            raw_path = self._record_path
            clean_writer = self._clean_record_wave
            clean_path = self._clean_record_path
            ab_original_writer = self._ab_clean_original_wave
            ab_rvc_writer = self._ab_clean_rvc_wave
            ab_original_path = self._ab_clean_original_path
            ab_rvc_path = self._ab_clean_rvc_path
            ab_stamp = str(
                self._record_session_stamp
            )

            self._record_wave = None
            self._record_path = None
            self._clean_record_wave = None
            self._clean_record_path = None
            self._ab_clean_original_wave = None
            self._ab_clean_rvc_wave = None
            self._ab_clean_original_path = None
            self._ab_clean_rvc_path = None

            if raw_writer is not None:
                with contextlib.suppress(Exception):
                    raw_writer.close()

            if clean_writer is not None:
                with contextlib.suppress(Exception):
                    clean_writer.close()

            if ab_original_writer is not None:
                with contextlib.suppress(Exception):
                    ab_original_writer.close()

            if ab_rvc_writer is not None:
                with contextlib.suppress(Exception):
                    ab_rvc_writer.close()

        if raw_path is not None:
            self._last_raw_record_path = raw_path
            self.log(
                f"RAW WAV 녹음 완료: {raw_path}"
            )

        if clean_path is not None:
            self._last_clean_record_path = clean_path
            self.log(
                f"CLEAN WAV 녹음 완료: {clean_path}"
            )

        if (
            ab_original_path is not None
            and ab_rvc_path is not None
        ):
            self._ab_last_clean_original_path = ab_original_path
            self._ab_last_clean_rvc_path = ab_rvc_path
            self.log(
                "[RVC A/B] CLEAN 2종 녹음 완료."
            )

            try:
                alignment = align_wav_to_reference(
                    reference_path=ab_original_path,
                    target_path=ab_rvc_path,
                    max_lag_ms=max(
                        600,
                        int(
                            self.realtime_rvc_block_ms
                            * 3
                        ),
                    ),
                    minimum_correlation=0.35,
                    fade_in_ms=5.0,
                )
                self._ab_clean_rvc_alignment = dict(
                    alignment
                )

                if alignment.get(
                    "applied"
                ):
                    self.log(
                        "[RVC A/B] CLEAN RVC 녹음 경계 자동 정렬: "
                        f"{float(alignment.get('lag_ms', 0.0)):.0f}ms trim / "
                        f"corr={float(alignment.get('correlation', 0.0)):.3f}"
                    )
                else:
                    self.log(
                        "[RVC A/B] CLEAN RVC 자동 정렬 생략: "
                        f"lag={float(alignment.get('lag_ms', 0.0)):.0f}ms / "
                        f"corr={float(alignment.get('correlation', 0.0)):.3f}"
                    )

            except Exception as exc:
                self._ab_clean_rvc_alignment = {
                    "applied": False,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
                self.log(
                    "[RVC A/B] CLEAN RVC 자동 정렬 실패: "
                    f"{type(exc).__name__}: {exc}"
                )

            self._start_ab_broadcast_render(
                clean_original_path=ab_original_path,
                clean_rvc_path=ab_rvc_path,
                parent_dir=ab_original_path.parent,
                stamp=ab_stamp,
            )

        return (
            raw_path
            or clean_path
            or ab_original_path
            or broadcast_path
        )

    def snapshot(self) -> dict:
        with self._stats_lock:
            connected = bool(
                self.client_connected
            )
            label = str(
                self.client_device_label
            )
            client_rate = int(
                self.client_sample_rate
            )
            packet_frames = int(
                self.packet_frames
            )
            peak = float(
                self.peak_dbfs
            )
            received_samples = int(
                self.received_samples
            )
            received_packets = int(
                self.received_packets
            )
            last_packet = float(
                self.last_packet_monotonic
            )

        available = int(
            self.ring.available_samples
        )

        now = time.monotonic()
        broadcast = self.broadcast_capture.snapshot()
        final_monitor = self.final_voice_monitor.snapshot()
        realtime_rvc = self._realtime_rvc_snapshot()

        return {
            "running": bool(self._running),
            "connected": connected,
            "device_label": label,
            "client_sample_rate": client_rate,
            "packet_frames": packet_frames,
            "peak_dbfs": peak,
            "received_samples": received_samples,
            "received_packets": received_packets,
            "last_packet_age_ms": (
                (now - last_packet) * 1000.0
                if last_packet > 0.0
                else None
            ),
            "buffer_ms": (
                available
                * 1000.0
                / float(self.sample_rate)
            ),
            "sample_rate": int(
                self.sample_rate
            ),
            "overflow_ms": (
                self.ring.overflow_samples
                * 1000.0
                / float(self.sample_rate)
            ),
            "underflow_ms": (
                self.ring.underflow_samples
                * 1000.0
                / float(self.sample_rate)
            ),
            "recording": (
                self._record_wave is not None
                or self._clean_record_wave is not None
                or self._ab_clean_original_wave is not None
                or self._ab_clean_rvc_wave is not None
                or bool(
                    broadcast["recording"]
                )
            ),
            "record_raw_enabled": bool(
                self.record_raw_copy
            ),
            "record_path": (
                str(self._record_path)
                if self._record_path
                else ""
            ),
            "clean_record_path": (
                str(self._clean_record_path)
                if self._clean_record_path
                else ""
            ),
            "last_raw_record_path": (
                str(self._last_raw_record_path)
                if self._last_raw_record_path
                else ""
            ),
            "last_clean_record_path": (
                str(self._last_clean_record_path)
                if self._last_clean_record_path
                else ""
            ),
            "record_session_stamp": str(
                self._record_session_stamp
            ),
            "rvc_ab_enabled": bool(
                self.rvc_ab_debug_enabled
            ),
            "rvc_ab_rendering": bool(
                self._ab_rendering
            ),
            "rvc_ab_state": str(
                self._ab_render_state
            ),
            "rvc_ab_error": str(
                self._ab_render_error
            ),
            "rvc_ab_clean_original_path": (
                str(
                    self._ab_last_clean_original_path
                )
                if self._ab_last_clean_original_path
                else ""
            ),
            "rvc_ab_clean_rvc_path": (
                str(
                    self._ab_last_clean_rvc_path
                )
                if self._ab_last_clean_rvc_path
                else ""
            ),
            "rvc_ab_broadcast_original_path": (
                str(
                    self._ab_last_broadcast_original_path
                )
                if self._ab_last_broadcast_original_path
                else ""
            ),
            "rvc_ab_broadcast_rvc_path": (
                str(
                    self._ab_last_broadcast_rvc_path
                )
                if self._ab_last_broadcast_rvc_path
                else ""
            ),
            "rvc_ab_report_path": (
                str(
                    self._ab_last_report_path
                )
                if self._ab_last_report_path
                else ""
            ),
            "rvc_ab_clean_rvc_alignment": dict(
                self._ab_clean_rvc_alignment
            ),
            "transient_hits": int(
                self.dsp.transient_hits
            ),
            "smart_gain_db": float(
                self.dsp.smart_gain_current_db
            ),
            "smart_gain_input_rms_db": float(
                self.dsp.smart_gain_last_input_rms_db
            ),
            "smart_gain_output_rms_db": float(
                self.dsp.smart_gain_last_output_rms_db
            ),
            "limiter_reduction_db": float(
                self.dsp.limiter_reduction_db
            ),
            "broadcast_recording": bool(
                broadcast["recording"]
            ),
            "broadcast_record_path": str(
                broadcast["path"]
            ),
            "broadcast_last_path": str(
                broadcast["last_path"]
            ),
            "broadcast_sample_rate": int(
                broadcast["sample_rate"]
            ),
            "broadcast_peak_dbfs": float(
                broadcast["peak_dbfs"]
            ),
            "broadcast_received_frames": int(
                broadcast["received_frames"]
            ),
            "broadcast_status": str(
                broadcast["status"]
            ),
            "final_monitor_running": bool(
                final_monitor["running"]
            ),
            "final_monitor_status": str(
                final_monitor["status"]
            ),
            "final_monitor_peak_dbfs": float(
                final_monitor["peak_dbfs"]
            ),
            "final_monitor_buffer_ms": float(
                final_monitor["buffer_ms"]
            ),
            "final_monitor_underflow_ms": float(
                final_monitor["underflow_ms"]
            ),
            "final_monitor_input_name": str(
                final_monitor["input_name"]
            ),
            "final_monitor_output_name": str(
                final_monitor["output_name"]
            ),
            "final_monitor_gain_db": float(
                final_monitor["gain_db"]
            ),
            "realtime_rvc": realtime_rvc,
        }


class PhoneMicBridgeWidget(QWidget):
    native_action_done = Signal(
        str,
        bool,
        str,
    )

    def __init__(
        self,
        *,
        settings: QSettings | None = None,
        parent=None,
    ) -> None:
        super().__init__(
            parent
        )

        self.settings = (
            settings
            if settings is not None
            else QSettings(
                "VocalPitchAnalyzer",
                "VocalPitchAnalyzer",
            )
        )

        self.runtime = PhoneMicRuntime()
        self.camera_controller = S24CameraController(
            video_port=VIDEO_WS_PORT,
            control_port=CAMERA_CONTROL_PORT,
            log_callback=self.runtime.log,
        )
        self._devices: list[tuple[int, str]] = []
        self._broadcast_input_devices: list[tuple[int, str]] = []
        self._final_monitor_output_devices: list[tuple[int, str]] = []
        self._camera_device_signature: tuple = ()
        self._camera_preview_seq = -1
        self._native_action_busy = False

        self.native_action_done.connect(
            self._on_native_action_done
        )

        self._build_ui()
        self.refresh_output_devices()
        self.refresh_adb_status()

        self.timer = QTimer(
            self
        )
        self.timer.setInterval(
            250
        )
        self.timer.timeout.connect(
            self._refresh_runtime_ui
        )
        self.timer.start()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(
            self
        )
        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.page_scroll = QScrollArea(
            self
        )
        self.page_scroll.setWidgetResizable(
            True
        )
        self.page_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        scroll_content = QWidget()
        scroll_content.setObjectName(
            "phoneMicScrollContent"
        )

        root = QVBoxLayout(
            scroll_content
        )
        root.setSizeConstraint(
            QLayout.SizeConstraint.SetMinimumSize
        )

        self.page_scroll.setWidget(
            scroll_content
        )
        outer.addWidget(
            self.page_scroll,
            1,
        )

        intro = QGroupBox(
            "Galaxy S24 Ultra AV Bridge v4.0"
        )
        intro_layout = QVBoxLayout(
            intro
        )

        text = QLabel(
            "S24 마이크 + 카메라를 하나의 USB/ADB 브리지에서 처리합니다. "
            "오디오는 Smart Gain/RVC/NVIDIA Broadcast로, 영상은 카메라 선택·미러·회전·줌·"
            "색보정·AI 배경 흐림/교체를 거쳐 선택적으로 Windows 가상 웹캠으로 출력합니다."
        )
        text.setWordWrap(
            True
        )
        intro_layout.addWidget(
            text
        )

        self.runtime_label = QLabel(
            runtime_status_text()
        )
        self.runtime_label.setWordWrap(
            True
        )
        self.runtime_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        intro_layout.addWidget(
            self.runtime_label
        )
        root.addWidget(
            intro
        )

        adb_group = QGroupBox(
            "1. Galaxy S24 Ultra USB 연결"
        )
        adb_layout = QVBoxLayout(
            adb_group
        )

        guide = QLabel(
            "휴대폰: 설정 → 휴대전화 정보 → 소프트웨어 정보 → "
            "'빌드번호' 7회 탭 → 개발자 옵션 → USB 디버깅 ON.\n"
            "Chrome 브리지: 마이크+카메라 테스트/설정용. Android가 화면을 잠그거나 "
            "Chrome 페이지를 freeze하면 AudioWorklet/WebSocket이 중단될 수 있어 화면 OFF용으로는 부적합합니다.\n"
            "Native Mic: Android microphone Foreground Service + PARTIAL_WAKE_LOCK으로 동작하므로 "
            "시작 후 화면을 끄거나 앱을 백그라운드로 보내도 마이크 전송을 계속하도록 만든 모드입니다."
        )
        guide.setWordWrap(
            True
        )
        adb_layout.addWidget(
            guide
        )

        adb_row = QHBoxLayout()

        self.adb_status_label = QLabel(
            "ADB 확인 중..."
        )
        self.adb_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        adb_row.addWidget(
            self.adb_status_label,
            1,
        )

        self.adb_refresh_button = QPushButton(
            "ADB 새로고침"
        )
        self.adb_refresh_button.clicked.connect(
            self.refresh_adb_status
        )
        adb_row.addWidget(
            self.adb_refresh_button
        )

        self.phone_open_button = QPushButton(
            "Chrome AV 브리지 열기"
        )
        self.phone_open_button.clicked.connect(
            self.open_phone_page
        )
        adb_row.addWidget(
            self.phone_open_button
        )

        adb_layout.addLayout(
            adb_row
        )

        native_row = QHBoxLayout()

        self.native_mic_status_label = QLabel(
            "Native Mic 상태 확인 중..."
        )
        self.native_mic_status_label.setWordWrap(
            True
        )
        self.native_mic_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        native_row.addWidget(
            self.native_mic_status_label,
            1,
        )

        self.native_mic_install_button = QPushButton(
            "Native Mic APK 설치"
        )
        self.native_mic_install_button.clicked.connect(
            self.install_native_mic
        )
        native_row.addWidget(
            self.native_mic_install_button
        )

        self.native_mic_launch_button = QPushButton(
            "Native Mic 실행 / 화면 OFF"
        )
        self.native_mic_launch_button.setToolTip(
            "adb reverse tcp:8791을 설정하고 Native Mic Activity를 엽니다. "
            "권한이 이미 있으면 microphone foreground service를 자동 시작합니다."
        )
        self.native_mic_launch_button.clicked.connect(
            self.launch_native_mic
        )
        native_row.addWidget(
            self.native_mic_launch_button
        )

        adb_layout.addLayout(
            native_row
        )

        native_note = QLabel(
            "APK가 없으면 'Native Mic APK 설치' 버튼이 자동으로 Android APK 빌드부터 시도합니다. "
            "최초 빌드는 Gradle/Android 구성 다운로드 때문에 시간이 걸릴 수 있습니다. "
            "앱에서 마이크 권한을 허용한 뒤 'Start native mic'을 누르면 지속 알림이 표시됩니다. "
            "그 알림이 떠 있는 동안에는 화면을 꺼도 마이크 서비스가 계속 동작합니다. "
            "삼성 배터리 절전이 강하게 적용되는 경우 앱 배터리 사용을 '제한 없음'으로 바꾸세요."
        )
        native_note.setWordWrap(
            True
        )
        adb_layout.addWidget(
            native_note
        )

        root.addWidget(
            adb_group
        )

        output_group = QGroupBox(
            "2. Windows 오디오 라우팅 / NVIDIA Broadcast"
        )
        output_layout = QFormLayout(
            output_group
        )

        output_row = QHBoxLayout()

        self.output_combo = QComboBox()
        output_row.addWidget(
            self.output_combo,
            1,
        )

        self.refresh_devices_button = QPushButton(
            "장치 새로고침"
        )
        self.refresh_devices_button.clicked.connect(
            self.refresh_output_devices
        )
        output_row.addWidget(
            self.refresh_devices_button
        )

        output_layout.addRow(
            "출력 장치",
            output_row,
        )

        self.output_enabled_check = QCheckBox(
            "선택한 Windows 출력으로 CLEAN 신호 전송"
        )
        self.output_enabled_check.setChecked(
            self.settings.value(
                "phone_mic_monitor_enabled_v35",
                False,
                type=bool,
            )
        )
        self.output_enabled_check.setToolTip(
            "NVIDIA Broadcast 사용 시 출력 장치를 CABLE Input으로 선택하고 ON으로 두세요. "
            "그러면 CLEAN 신호가 VB-CABLE을 통해 NVIDIA Broadcast로 전달됩니다."
        )
        self.output_enabled_check.toggled.connect(
            self.on_monitor_toggled
        )
        output_layout.addRow(
            "",
            self.output_enabled_check,
        )

        self.broadcast_record_check = QCheckBox(
            "NVIDIA Broadcast 최종 보정음 동시 녹음"
        )
        self.broadcast_record_check.setChecked(
            self.settings.value(
                "phone_mic_broadcast_record_enabled",
                True,
                type=bool,
            )
        )

        self.broadcast_input_combo = QComboBox()
        self.broadcast_input_combo.setToolTip(
            "보통 'Microphone (NVIDIA Broadcast)' 또는 "
            "'마이크(NVIDIA Broadcast)'를 선택합니다."
        )
        self.broadcast_input_combo.currentIndexChanged.connect(
            self.on_final_monitor_device_changed
        )

        broadcast_row = QHBoxLayout()
        broadcast_row.addWidget(
            self.broadcast_record_check
        )
        broadcast_row.addWidget(
            self.broadcast_input_combo,
            1,
        )
        output_layout.addRow(
            "최종 마이크 녹음",
            broadcast_row,
        )

        broadcast_note = QLabel(
            "권장: 우리 앱 출력=CABLE Input → NVIDIA Broadcast 입력=CABLE Output → "
            "최종 마이크 녹음=Microphone (NVIDIA Broadcast). "
            "Broadcast 처리 지연 때문에 RAW/CLEAN과 시작 파형은 조금 어긋날 수 있습니다."
        )
        broadcast_note.setWordWrap(
            True
        )
        output_layout.addRow(
            broadcast_note
        )

        self.final_monitor_check = QCheckBox(
            "상대방에게 전달되는 최종 보이스를 내 PC에서도 듣기"
        )
        self.final_monitor_check.setChecked(
            self.settings.value(
                "phone_mic_final_monitor_enabled_v45",
                False,
                type=bool,
            )
        )
        self.final_monitor_check.setToolTip(
            "Microphone (NVIDIA Broadcast) 최종 출력을 다시 캡처해 "
            "선택한 실제 스피커/헤드폰으로 재생합니다."
        )
        self.final_monitor_check.toggled.connect(
            self.on_final_monitor_toggled
        )
        output_layout.addRow(
            "최종 보이스 모니터",
            self.final_monitor_check,
        )

        final_monitor_row = QHBoxLayout()

        self.final_monitor_output_combo = QComboBox()
        self.final_monitor_output_combo.setToolTip(
            "최종 보이스를 들을 실제 스피커/헤드폰을 선택하세요. "
            "CABLE Input을 선택하지 않는 것을 권장합니다."
        )
        self.final_monitor_output_combo.currentIndexChanged.connect(
            self.on_final_monitor_device_changed
        )
        final_monitor_row.addWidget(
            self.final_monitor_output_combo,
            1,
        )

        self.final_monitor_gain_spin = QDoubleSpinBox()
        self.final_monitor_gain_spin.setRange(
            -40.0,
            12.0,
        )
        self.final_monitor_gain_spin.setDecimals(1)
        self.final_monitor_gain_spin.setSingleStep(1.0)
        self.final_monitor_gain_spin.setSuffix(" dB")
        self.final_monitor_gain_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_final_monitor_gain_db_v45",
                    -6.0,
                )
            )
        )
        self.final_monitor_gain_spin.setToolTip(
            "내가 듣는 모니터 볼륨만 조절합니다. "
            "Discord/게임으로 전달되는 마이크 볼륨에는 영향을 주지 않습니다."
        )
        self.final_monitor_gain_spin.valueChanged.connect(
            self.on_final_monitor_gain_changed
        )
        final_monitor_row.addWidget(
            self.final_monitor_gain_spin
        )

        output_layout.addRow(
            "내 모니터 출력",
            final_monitor_row,
        )

        final_monitor_note = QLabel(
            "경로: Microphone (NVIDIA Broadcast) → 내 스피커/헤드폰. "
            "즉 CLEAN/RVC가 아니라 상대방에게 전달되는 최종 보이스를 그대로 듣습니다. "
            "스피커 사용 시 S24 마이크가 다시 소리를 받아 에코/하울링이 생길 수 있으므로 "
            "가능하면 헤드폰/이어폰을 권장합니다."
        )
        final_monitor_note.setWordWrap(True)
        output_layout.addRow(
            final_monitor_note
        )

        self.jitter_spin = QSpinBox()
        self.jitter_spin.setRange(
            20,
            500,
        )
        self.jitter_spin.setSuffix(
            " ms"
        )
        self.jitter_spin.setValue(
            int(
                self.settings.value(
                    "phone_mic_jitter_ms",
                    DEFAULT_JITTER_MS,
                )
            )
        )
        self.jitter_spin.setToolTip(
            "낮을수록 지연은 줄지만 끊김에 민감합니다. "
            "USB에서는 60~100ms부터 시작하는 것을 권장합니다."
        )
        output_layout.addRow(
            "Jitter Buffer",
            self.jitter_spin,
        )

        root.addWidget(
            output_group
        )

        dsp_group = QGroupBox(
            "3. Noise / DSP"
        )
        dsp_layout = QFormLayout(
            dsp_group
        )

        raw_note = QLabel(
            "RAW WAV는 PC DSP/Smart Gain 전 신호를 그대로 보존합니다. "
            "CLEAN WAV에는 Gain/HPF/Gate/순간음 억제/Smart Voice Gain/Limiter가 적용됩니다. "
            "책상 거치 실사용에서는 CLEAN 파일을 바로 쓰고, RAW는 원본 보관용으로 남길 수 있습니다. "
            "휴대폰 Browser Noise Suppression은 PC에 오기 전에 적용되므로 RAW에도 반영됩니다."
        )
        raw_note.setWordWrap(
            True
        )
        dsp_layout.addRow(
            raw_note
        )

        self.s24_voice_focus_dsp_button = QPushButton(
            "S24 Voice Focus DSP 권장값"
        )
        self.s24_voice_focus_dsp_button.setToolTip(
            "S24 내장 마이크용 보수적 프리셋입니다. "
            "80Hz 저역 컷, 순간음 억제, Smart Voice Gain, limiter를 켜고 "
            "Hard Gate는 사용하지 않습니다. 각 값은 적용 후 다시 개별 조절할 수 있습니다."
        )
        self.s24_voice_focus_dsp_button.clicked.connect(
            self.apply_s24_voice_focus_dsp_preset
        )
        dsp_layout.addRow(
            "",
            self.s24_voice_focus_dsp_button,
        )

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(
            -24.0,
            24.0,
        )
        self.gain_spin.setDecimals(
            1
        )
        self.gain_spin.setSingleStep(
            0.5
        )
        self.gain_spin.setSuffix(
            " dB"
        )
        self.gain_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_gain_db",
                    0.0,
                )
            )
        )
        dsp_layout.addRow(
            "Gain",
            self.gain_spin,
        )

        self.highpass_check = QCheckBox(
            "High-pass 사용"
        )
        self.highpass_check.setChecked(
            self.settings.value(
                "phone_mic_highpass_enabled",
                False,
                type=bool,
            )
        )

        self.highpass_spin = QDoubleSpinBox()
        self.highpass_spin.setRange(
            20.0,
            250.0,
        )
        self.highpass_spin.setSuffix(
            " Hz"
        )
        self.highpass_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_highpass_hz",
                    70.0,
                )
            )
        )

        hp_row = QHBoxLayout()
        hp_row.addWidget(
            self.highpass_check
        )
        hp_row.addWidget(
            self.highpass_spin
        )
        dsp_layout.addRow(
            "저역 컷",
            hp_row,
        )

        self.gate_check = QCheckBox(
            "Soft Gate/Expander 사용"
        )
        self.gate_check.setChecked(
            self.settings.value(
                "phone_mic_gate_enabled",
                False,
                type=bool,
            )
        )

        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setRange(
            -90.0,
            -10.0,
        )
        self.gate_spin.setSuffix(
            " dBFS"
        )
        self.gate_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_gate_threshold_db",
                    -55.0,
                )
            )
        )

        gate_row = QHBoxLayout()
        gate_row.addWidget(
            self.gate_check
        )
        gate_row.addWidget(
            self.gate_spin
        )
        dsp_layout.addRow(
            "Gate",
            gate_row,
        )

        self.transient_check = QCheckBox(
            "키보드/마우스 순간음 억제"
        )
        self.transient_check.setChecked(
            self.settings.value(
                "phone_mic_transient_suppression_enabled",
                True,
                type=bool,
            )
        )
        self.transient_check.setToolTip(
            "말이 없는 저레벨 구간에서 crest factor와 고주파 중심이 큰 "
            "짧은 클릭/키보드 충격음을 보수적으로 줄입니다. "
            "발음 손상을 피하기 위해 강한 음성 프레임에는 적용하지 않습니다."
        )

        self.transient_reduction_spin = QDoubleSpinBox()
        self.transient_reduction_spin.setRange(
            3.0,
            24.0,
        )
        self.transient_reduction_spin.setDecimals(
            1
        )
        self.transient_reduction_spin.setSingleStep(
            1.0
        )
        self.transient_reduction_spin.setSuffix(
            " dB"
        )
        self.transient_reduction_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_transient_reduction_db",
                    10.0,
                )
            )
        )

        transient_row = QHBoxLayout()
        transient_row.addWidget(
            self.transient_check
        )
        transient_row.addWidget(
            self.transient_reduction_spin
        )
        dsp_layout.addRow(
            "Click/Key",
            transient_row,
        )

        self.smart_gain_check = QCheckBox(
            "Smart Voice Gain 사용 (책상 거리 음성 증폭, 권장)"
        )
        self.smart_gain_check.setChecked(
            self.settings.value(
                "phone_mic_smart_gain_enabled",
                True,
                type=bool,
            )
        )
        self.smart_gain_check.setToolTip(
            "말소리의 단기 RMS를 목표 레벨 쪽으로 천천히 올리고, "
            "가까이 말해서 갑자기 커질 때는 빠르게 Gain을 줄입니다. "
            "무음 구간은 목표 레벨까지 끌어올리지 않아 배경 소음 펌핑을 줄입니다."
        )

        self.smart_gain_target_spin = QDoubleSpinBox()
        self.smart_gain_target_spin.setRange(
            -30.0,
            -12.0,
        )
        self.smart_gain_target_spin.setDecimals(
            1
        )
        self.smart_gain_target_spin.setSingleStep(
            1.0
        )
        self.smart_gain_target_spin.setSuffix(
            " dBFS RMS"
        )
        self.smart_gain_target_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_smart_gain_target_db",
                    -20.0,
                )
            )
        )

        self.smart_gain_max_spin = QDoubleSpinBox()
        self.smart_gain_max_spin.setRange(
            0.0,
            30.0,
        )
        self.smart_gain_max_spin.setDecimals(
            1
        )
        self.smart_gain_max_spin.setSingleStep(
            1.0
        )
        self.smart_gain_max_spin.setSuffix(
            " dB"
        )
        self.smart_gain_max_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_smart_gain_max_boost_db",
                    18.0,
                )
            )
        )

        smart_gain_row = QHBoxLayout()
        smart_gain_row.addWidget(
            self.smart_gain_check
        )
        smart_gain_row.addWidget(
            QLabel(
                "Target"
            )
        )
        smart_gain_row.addWidget(
            self.smart_gain_target_spin
        )
        smart_gain_row.addWidget(
            QLabel(
                "Max"
            )
        )
        smart_gain_row.addWidget(
            self.smart_gain_max_spin
        )
        dsp_layout.addRow(
            "Voice Level",
            smart_gain_row,
        )

        self.limiter_check = QCheckBox(
            "Limiter 사용"
        )
        self.limiter_check.setChecked(
            self.settings.value(
                "phone_mic_limiter_enabled",
                True,
                type=bool,
            )
        )

        self.limiter_ceiling_spin = QDoubleSpinBox()
        self.limiter_ceiling_spin.setRange(
            -6.0,
            -0.1,
        )
        self.limiter_ceiling_spin.setDecimals(
            1
        )
        self.limiter_ceiling_spin.setSingleStep(
            0.5
        )
        self.limiter_ceiling_spin.setSuffix(
            " dBFS"
        )
        self.limiter_ceiling_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_limiter_ceiling_db",
                    -1.0,
                )
            )
        )

        limiter_row = QHBoxLayout()
        limiter_row.addWidget(
            self.limiter_check
        )
        limiter_row.addWidget(
            self.limiter_ceiling_spin
        )
        dsp_layout.addRow(
            "Peak Safety",
            limiter_row,
        )

        self.raw_record_check = QCheckBox(
            "RAW 입력 WAV 저장"
        )
        self.raw_record_check.setChecked(
            self.settings.value(
                "phone_mic_record_raw_copy",
                False,
                type=bool,
            )
        )
        self.raw_record_check.setToolTip(
            "PC DSP/Smart Voice Gain 전 입력 신호를 s24_raw_*.wav로 저장합니다. "
            "평소에는 OFF를 권장하고, 필터 비교/디버그/원본 보관이 필요할 때만 켜세요."
        )
        dsp_layout.addRow(
            "",
            self.raw_record_check,
        )

        self.clean_record_check = QCheckBox(
            "CLEAN WAV 저장 (PC DSP + Smart Gain 적용)"
        )
        self.clean_record_check.setChecked(
            self.settings.value(
                "phone_mic_record_clean_copy",
                True,
                type=bool,
            )
        )
        self.clean_record_check.setToolTip(
            "녹음 시 s24_raw_*.wav와 s24_clean_*.wav를 동시에 저장합니다. "
            "RAW는 보존용, CLEAN은 Gain/HPF/Gate/키보드·마우스 순간음 억제 적용본입니다."
        )
        dsp_layout.addRow(
            "",
            self.clean_record_check,
        )

        root.addWidget(
            dsp_group
        )

        camera_group = QGroupBox(
            "4. S24 Camera / Virtual Webcam"
        )
        camera_layout = QVBoxLayout(camera_group)
        self.camera_runtime_label = QLabel(camera_runtime_status_text())
        self.camera_runtime_label.setWordWrap(True)
        camera_layout.addWidget(self.camera_runtime_label)

        self.camera_preview = QLabel("카메라 Preview")
        self.camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setMinimumSize(480, 270)
        self.camera_preview.setStyleSheet("background:#050608; border:1px solid #30343b;")
        camera_layout.addWidget(self.camera_preview)

        camera_form = QFormLayout()
        self.camera_device_combo = QComboBox()
        self.camera_device_combo.addItem("휴대폰 카메라 목록 대기 중", "")
        camera_form.addRow("Phone Camera", self.camera_device_combo)

        self.camera_resolution_combo = QComboBox()
        for label, size in (("640 x 480", (640,480)), ("1280 x 720 (권장)",(1280,720)), ("1920 x 1080",(1920,1080))):
            self.camera_resolution_combo.addItem(label, size)
        saved_res=self.settings.value("s24_camera_resolution","1280x720",type=str)
        self.camera_resolution_combo.setCurrentIndex({"640x480":0,"1280x720":1,"1920x1080":2}.get(saved_res,1))
        camera_form.addRow("Capture", self.camera_resolution_combo)

        self.camera_fps_combo = QComboBox()
        for fps in (10,15,20,30): self.camera_fps_combo.addItem(f"{fps} fps",fps)
        idx=self.camera_fps_combo.findData(int(self.settings.value("s24_camera_fps",15)))
        self.camera_fps_combo.setCurrentIndex(idx if idx>=0 else 1)
        camera_form.addRow("FPS", self.camera_fps_combo)

        self.camera_quality_spin=QDoubleSpinBox(); self.camera_quality_spin.setRange(.30,.95); self.camera_quality_spin.setDecimals(2); self.camera_quality_spin.setSingleStep(.05); self.camera_quality_spin.setValue(float(self.settings.value("s24_camera_jpeg_quality",.72)))
        camera_form.addRow("USB JPEG Quality",self.camera_quality_spin)

        row=QHBoxLayout()
        self.camera_mirror_check=QCheckBox("미러"); self.camera_mirror_check.setChecked(self.settings.value("s24_camera_mirror",True,type=bool)); row.addWidget(self.camera_mirror_check)
        self.camera_rotation_combo=QComboBox()
        for v in (0,90,180,270): self.camera_rotation_combo.addItem(f"{v}°",v)
        ridx=self.camera_rotation_combo.findData(int(self.settings.value("s24_camera_rotation",0))); self.camera_rotation_combo.setCurrentIndex(ridx if ridx>=0 else 0); row.addWidget(self.camera_rotation_combo)
        self.camera_zoom_spin=QDoubleSpinBox(); self.camera_zoom_spin.setRange(1,4); self.camera_zoom_spin.setDecimals(2); self.camera_zoom_spin.setSingleStep(.1); self.camera_zoom_spin.setSuffix(" x"); self.camera_zoom_spin.setValue(float(self.settings.value("s24_camera_zoom",1.0))); row.addWidget(self.camera_zoom_spin)
        camera_form.addRow("Transform",row)

        self.camera_brightness_spin=QSpinBox(); self.camera_brightness_spin.setRange(-100,100); self.camera_brightness_spin.setValue(int(self.settings.value("s24_camera_brightness",0))); camera_form.addRow("Brightness",self.camera_brightness_spin)
        self.camera_contrast_spin=QDoubleSpinBox(); self.camera_contrast_spin.setRange(0,3); self.camera_contrast_spin.setDecimals(2); self.camera_contrast_spin.setSingleStep(.05); self.camera_contrast_spin.setValue(float(self.settings.value("s24_camera_contrast",1.0))); camera_form.addRow("Contrast",self.camera_contrast_spin)
        self.camera_saturation_spin=QDoubleSpinBox(); self.camera_saturation_spin.setRange(0,3); self.camera_saturation_spin.setDecimals(2); self.camera_saturation_spin.setSingleStep(.05); self.camera_saturation_spin.setValue(float(self.settings.value("s24_camera_saturation",1.0))); camera_form.addRow("Saturation",self.camera_saturation_spin)

        self.camera_background_combo=QComboBox(); self.camera_background_combo.addItem("배경 처리 없음","none"); self.camera_background_combo.addItem("AI 배경 흐림","blur"); self.camera_background_combo.addItem("AI 가상 배경 이미지","image")
        bidx=self.camera_background_combo.findData(self.settings.value("s24_camera_background_mode","none",type=str)); self.camera_background_combo.setCurrentIndex(bidx if bidx>=0 else 0); camera_form.addRow("Background",self.camera_background_combo)
        self.camera_background_path=self.settings.value("s24_camera_background_image","",type=str)
        bgrow=QHBoxLayout(); self.camera_background_label=QLabel(self.camera_background_path or "배경 이미지 미선택"); self.camera_background_label.setWordWrap(True); bgrow.addWidget(self.camera_background_label,1); self.camera_background_button=QPushButton("배경 이미지 선택"); self.camera_background_button.clicked.connect(self.choose_camera_background); bgrow.addWidget(self.camera_background_button); camera_form.addRow("Virtual BG Image",bgrow)
        self.camera_blur_spin=QSpinBox(); self.camera_blur_spin.setRange(3,60); self.camera_blur_spin.setValue(int(self.settings.value("s24_camera_background_blur",25))); camera_form.addRow("Background Blur",self.camera_blur_spin)
        self.camera_virtual_check=QCheckBox("Windows 가상 웹캠 출력 (OBS Virtual Camera driver 필요)"); self.camera_virtual_check.setChecked(self.settings.value("s24_camera_virtual_enabled",False,type=bool)); camera_form.addRow("",self.camera_virtual_check)
        camera_layout.addLayout(camera_form)

        brow=QHBoxLayout()
        self.camera_worker_button=QPushButton("카메라 워커 시작"); self.camera_worker_button.clicked.connect(self.start_camera_worker); brow.addWidget(self.camera_worker_button)
        self.camera_start_button=QPushButton("설정 적용 / 휴대폰 카메라 시작·재시작"); self.camera_start_button.clicked.connect(self.start_phone_camera); brow.addWidget(self.camera_start_button)
        self.camera_stop_button=QPushButton("카메라 중지"); self.camera_stop_button.clicked.connect(self.stop_phone_camera); brow.addWidget(self.camera_stop_button); camera_layout.addLayout(brow)
        self.camera_live_label=QLabel("Camera worker: stopped"); self.camera_live_label.setWordWrap(True); camera_layout.addWidget(self.camera_live_label)
        note=QLabel("처음에는 카메라 워커 시작 → 휴대폰 페이지에서 카메라 권한 허용/PC 제어 연결을 한 번 누르세요. 이후 카메라/해상도/FPS와 영상 효과는 PC에서 제어합니다."); note.setWordWrap(True); camera_layout.addWidget(note)
        root.addWidget(camera_group)

        rvc_group = QGroupBox(
            "5. 실시간 RVC Voice Changer (Experimental)"
        )
        rvc_layout = QFormLayout(
            rvc_group
        )

        self.realtime_rvc_status_label = QLabel(
            realtime_rvc_status_text()
        )
        self.realtime_rvc_status_label.setWordWrap(
            True
        )
        self.realtime_rvc_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rvc_layout.addRow(
            "Runtime",
            self.realtime_rvc_status_label,
        )

        self.realtime_rvc_enable_check = QCheckBox(
            "실시간 RVC 사용"
        )
        self.realtime_rvc_enable_check.setChecked(
            self.settings.value(
                "phone_mic_realtime_rvc_enabled",
                False,
                type=bool,
            )
        )
        self.realtime_rvc_enable_check.setToolTip(
            "ON + 엔진 READY일 때 CLEAN 음성이 RVC를 거쳐 CABLE Input으로 전달됩니다. "
            "엔진 로딩/실패 중에는 자동으로 CLEAN 원음을 bypass합니다."
        )
        rvc_layout.addRow(
            "",
            self.realtime_rvc_enable_check,
        )

        self.rvc_ab_debug_check = QCheckBox(
            "RVC A/B 디버그 4종 비교 저장"
        )
        self.rvc_ab_debug_check.setChecked(
            self.settings.value(
                "phone_mic_rvc_ab_debug_enabled",
                False,
                type=bool,
            )
        )
        self.rvc_ab_debug_check.setToolTip(
            "테스트용. 녹음 중 CLEAN 원본과 RVC 적용본을 동시에 저장하고, "
            "녹음 종료 후 두 파일을 NVIDIA Broadcast에 순차 재생해 "
            "Broadcast 원본/RVC 파일까지 총 4종을 만듭니다."
        )
        rvc_layout.addRow(
            "",
            self.rvc_ab_debug_check,
        )

        self.rvc_ab_status_label = QLabel(
            "A/B: idle"
        )
        self.rvc_ab_status_label.setWordWrap(
            True
        )
        self.rvc_ab_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rvc_layout.addRow(
            "A/B Debug",
            self.rvc_ab_status_label,
        )

        saved_rt_model = self.settings.value(
            "phone_mic_realtime_rvc_model",
            self.settings.value(
                "rvc_model_path",
                "",
                type=str,
            ),
            type=str,
        )
        saved_rt_index = self.settings.value(
            "phone_mic_realtime_rvc_index",
            self.settings.value(
                "rvc_index_path",
                "",
                type=str,
            ),
            type=str,
        )

        self.realtime_rvc_model_path = str(
            saved_rt_model
            or ""
        )
        self.realtime_rvc_index_path = str(
            saved_rt_index
            or ""
        )

        model_row = QHBoxLayout()
        self.realtime_rvc_model_label = QLabel(
            self.realtime_rvc_model_path
            or "RVC .pth 모델을 선택하세요."
        )
        self.realtime_rvc_model_label.setWordWrap(
            True
        )
        self.realtime_rvc_model_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.realtime_rvc_model_button = QPushButton(
            ".pth 선택"
        )
        self.realtime_rvc_model_button.clicked.connect(
            self.choose_realtime_rvc_model
        )
        model_row.addWidget(
            self.realtime_rvc_model_label,
            1,
        )
        model_row.addWidget(
            self.realtime_rvc_model_button
        )
        rvc_layout.addRow(
            "Model",
            model_row,
        )

        index_row = QHBoxLayout()
        self.realtime_rvc_index_label = QLabel(
            self.realtime_rvc_index_path
            or "Index 미선택 - Index Rate 0으로 동작"
        )
        self.realtime_rvc_index_label.setWordWrap(
            True
        )
        self.realtime_rvc_index_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.realtime_rvc_index_button = QPushButton(
            ".index 선택"
        )
        self.realtime_rvc_index_button.clicked.connect(
            self.choose_realtime_rvc_index
        )
        self.realtime_rvc_index_clear_button = QPushButton(
            "Index 지우기"
        )
        self.realtime_rvc_index_clear_button.clicked.connect(
            self.clear_realtime_rvc_index
        )
        index_row.addWidget(
            self.realtime_rvc_index_label,
            1,
        )
        index_row.addWidget(
            self.realtime_rvc_index_button
        )
        index_row.addWidget(
            self.realtime_rvc_index_clear_button
        )
        rvc_layout.addRow(
            "Feature Index",
            index_row,
        )

        self.realtime_rvc_pitch_spin = QSpinBox()
        self.realtime_rvc_pitch_spin.setRange(
            -24,
            24,
        )
        self.realtime_rvc_pitch_spin.setSuffix(
            " semitone"
        )
        self.realtime_rvc_pitch_spin.setValue(
            int(
                self.settings.value(
                    "phone_mic_realtime_rvc_pitch",
                    0,
                )
            )
        )
        rvc_layout.addRow(
            "Pitch",
            self.realtime_rvc_pitch_spin,
        )

        self.realtime_rvc_index_rate_spin = QDoubleSpinBox()
        self.realtime_rvc_index_rate_spin.setRange(
            0.0,
            1.0,
        )
        self.realtime_rvc_index_rate_spin.setDecimals(
            2
        )
        self.realtime_rvc_index_rate_spin.setSingleStep(
            0.05
        )
        self.realtime_rvc_index_rate_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_realtime_rvc_index_rate",
                    0.35,
                )
            )
        )
        self.realtime_rvc_index_rate_spin.setToolTip(
            "실시간 대화는 0.30~0.50부터 권장. "
            "Index가 없으면 자동으로 0 처리됩니다."
        )
        rvc_layout.addRow(
            "Index Rate",
            self.realtime_rvc_index_rate_spin,
        )

        self.realtime_rvc_f0_guard_check = QCheckBox(
            "F0 Stability Guard"
        )
        self.realtime_rvc_f0_guard_check.setChecked(
            self.settings.value(
                "phone_mic_realtime_rvc_f0_guard_enabled",
                True,
                type=bool,
            )
        )
        self.realtime_rvc_f0_guard_check.setToolTip(
            "RMVPE의 짧은 오검출/무음 구간 F0 확산을 억제합니다. "
            "특히 upstream 방식의 leading/trailing F0 extrapolation을 막습니다."
        )
        rvc_layout.addRow(
            "",
            self.realtime_rvc_f0_guard_check,
        )

        self.realtime_rvc_f0_diagnostic_check = QCheckBox(
            "F0 진단 로그 저장"
        )
        self.realtime_rvc_f0_diagnostic_check.setChecked(
            self.settings.value(
                "phone_mic_realtime_rvc_f0_diagnostic_enabled",
                True,
                type=bool,
            )
        )
        self.realtime_rvc_f0_diagnostic_check.setToolTip(
            "logs/realtime_rvc_f0_last.csv 와 .json에 "
            "입력 RMS, raw/guarded F0, voiced ratio, guard reason을 기록합니다."
        )
        rvc_layout.addRow(
            "",
            self.realtime_rvc_f0_diagnostic_check,
        )

        self.realtime_rvc_rmvpe_threshold_spin = QDoubleSpinBox()
        self.realtime_rvc_rmvpe_threshold_spin.setRange(
            0.01,
            0.30,
        )
        self.realtime_rvc_rmvpe_threshold_spin.setDecimals(
            3
        )
        self.realtime_rvc_rmvpe_threshold_spin.setSingleStep(
            0.01
        )
        self.realtime_rvc_rmvpe_threshold_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_realtime_rvc_rmvpe_threshold",
                    0.05,
                )
            )
        )
        self.realtime_rvc_rmvpe_threshold_spin.setToolTip(
            "기존 RVC realtime 기본은 0.03. "
            "실시간 마이크에서는 0.05부터 시작해 약한 잡음 F0 오검출을 줄입니다."
        )
        rvc_layout.addRow(
            "RMVPE Threshold",
            self.realtime_rvc_rmvpe_threshold_spin,
        )

        self.realtime_rvc_f0_quiet_rms_spin = QDoubleSpinBox()
        self.realtime_rvc_f0_quiet_rms_spin.setRange(
            -80.0,
            -20.0,
        )
        self.realtime_rvc_f0_quiet_rms_spin.setDecimals(
            1
        )
        self.realtime_rvc_f0_quiet_rms_spin.setSingleStep(
            1.0
        )
        self.realtime_rvc_f0_quiet_rms_spin.setSuffix(
            " dBFS"
        )
        self.realtime_rvc_f0_quiet_rms_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_realtime_rvc_f0_quiet_rms_db",
                    -48.0,
                )
            )
        )
        self.realtime_rvc_f0_quiet_rms_spin.setToolTip(
            "이 값 이하의 현재 CLEAN block은 F0를 강제로 unvoiced 처리합니다."
        )
        rvc_layout.addRow(
            "F0 Quiet RMS",
            self.realtime_rvc_f0_quiet_rms_spin,
        )

        self.realtime_rvc_f0_min_voiced_spin = QDoubleSpinBox()
        self.realtime_rvc_f0_min_voiced_spin.setRange(
            0.0,
            1.0,
        )
        self.realtime_rvc_f0_min_voiced_spin.setDecimals(
            2
        )
        self.realtime_rvc_f0_min_voiced_spin.setSingleStep(
            0.05
        )
        self.realtime_rvc_f0_min_voiced_spin.setValue(
            float(
                self.settings.value(
                    "phone_mic_realtime_rvc_f0_min_voiced_ratio",
                    0.15,
                )
            )
        )
        self.realtime_rvc_f0_min_voiced_spin.setToolTip(
            "약한 입력에서 raw voiced ratio가 이 값보다 낮으면 "
            "sparse F0 오검출로 판단해 현재 block F0를 제거합니다."
        )
        rvc_layout.addRow(
            "Min Voiced Ratio",
            self.realtime_rvc_f0_min_voiced_spin,
        )

        f0_time_row = QHBoxLayout()

        self.realtime_rvc_f0_max_gap_spin = QSpinBox()
        self.realtime_rvc_f0_max_gap_spin.setRange(
            0,
            200,
        )
        self.realtime_rvc_f0_max_gap_spin.setSuffix(
            " ms gap"
        )
        self.realtime_rvc_f0_max_gap_spin.setValue(
            int(
                self.settings.value(
                    "phone_mic_realtime_rvc_f0_max_gap_ms",
                    30,
                )
            )
        )
        f0_time_row.addWidget(
            self.realtime_rvc_f0_max_gap_spin
        )

        self.realtime_rvc_f0_min_run_spin = QSpinBox()
        self.realtime_rvc_f0_min_run_spin.setRange(
            0,
            200,
        )
        self.realtime_rvc_f0_min_run_spin.setSuffix(
            " ms min run"
        )
        self.realtime_rvc_f0_min_run_spin.setValue(
            int(
                self.settings.value(
                    "phone_mic_realtime_rvc_f0_min_run_ms",
                    40,
                )
            )
        )
        f0_time_row.addWidget(
            self.realtime_rvc_f0_min_run_spin
        )

        rvc_layout.addRow(
            "F0 Gap / Island",
            f0_time_row,
        )

        self.realtime_rvc_f0_preset_button = QPushButton(
            "F0 Guard 균형값 복원"
        )
        self.realtime_rvc_f0_preset_button.clicked.connect(
            self.apply_realtime_rvc_f0_guard_balanced_preset
        )
        rvc_layout.addRow(
            "",
            self.realtime_rvc_f0_preset_button,
        )

        self.realtime_rvc_block_combo = QComboBox()
        for label, value in (
            (
                "100 ms - 초저지연 / RTX 50 권장 테스트",
                100,
            ),
            (
                "120 ms - 저지연",
                120,
            ),
            (
                "150 ms - 균형",
                150,
            ),
            (
                "200 ms - 안정성 우선",
                200,
            ),
            (
                "250 ms - 높은 안정성",
                250,
            ),
            (
                "300 ms - 지연 큼",
                300,
            ),
        ):
            self.realtime_rvc_block_combo.addItem(
                label,
                value,
            )

        saved_block = int(
            self.settings.value(
                "phone_mic_realtime_rvc_block_ms",
                200,
            )
        )
        block_index = self.realtime_rvc_block_combo.findData(
            saved_block
        )
        self.realtime_rvc_block_combo.setCurrentIndex(
            block_index
            if block_index >= 0
            else max(
                0,
                self.realtime_rvc_block_combo.findData(
                    200
                ),
            )
        )
        rvc_layout.addRow(
            "Processing Block",
            self.realtime_rvc_block_combo,
        )

        rvc_button_row = QHBoxLayout()
        self.realtime_rvc_low_latency_button = QPushButton(
            "저지연 프리셋"
        )
        self.realtime_rvc_low_latency_button.setToolTip(
            "RVC Block=100ms / Jitter=30ms로 설정합니다. "
            "RTX 5070 Ti에서 우선 테스트하고 worker roundtrip이 90ms 이상이면 "
            "120~150ms로 올리세요. 변경 후 RVC 엔진 재시작이 필요합니다."
        )
        self.realtime_rvc_low_latency_button.clicked.connect(
            self.apply_realtime_rvc_low_latency_preset
        )
        rvc_button_row.addWidget(
            self.realtime_rvc_low_latency_button
        )

        self.realtime_rvc_start_button = QPushButton(
            "RVC 엔진 로드 / 재시작"
        )
        self.realtime_rvc_start_button.clicked.connect(
            self.start_realtime_rvc
        )
        self.realtime_rvc_stop_button = QPushButton(
            "RVC 엔진 중지"
        )
        self.realtime_rvc_stop_button.clicked.connect(
            self.stop_realtime_rvc
        )
        rvc_button_row.addWidget(
            self.realtime_rvc_start_button
        )
        rvc_button_row.addWidget(
            self.realtime_rvc_stop_button
        )
        rvc_layout.addRow(
            "",
            rvc_button_row,
        )

        self.realtime_rvc_live_label = QLabel(
            "State: stopped"
        )
        self.realtime_rvc_live_label.setWordWrap(
            True
        )
        self.realtime_rvc_live_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rvc_layout.addRow(
            "Live",
            self.realtime_rvc_live_label,
        )

        rvc_note = QLabel(
            "권장 체인: S24 → Smart Voice Gain → Realtime RVC → "
            "CABLE Input → NVIDIA Broadcast → 게임. "
            "모델 로딩 중/실패 시에는 CLEAN 원음이 자동 통과합니다. "
            "RMVPE + SOLA / eager CUDA 경로를 사용합니다. v3.9b는 added IVF index의 nprobe=1 sparse-search 문제를 런타임에서 보정합니다. "
            "F0 Stability Guard는 RMVPE raw F0를 먼저 기록하고, 짧은 voiced island를 제거하며 짧은 내부 gap만 보간합니다. 특히 upstream 방식처럼 첫/끝 무음까지 한 번의 F0로 extrapolation하지 않습니다. A/B 디버그는 녹음 경계도 자동 정렬합니다."
        )
        rvc_note.setWordWrap(
            True
        )
        rvc_layout.addRow(
            rvc_note
        )

        root.addWidget(
            rvc_group
        )

        control_group = QGroupBox(
            "6. 브리지 / 녹음"
        )
        control_layout = QVBoxLayout(
            control_group
        )

        button_row = QHBoxLayout()

        self.start_button = QPushButton(
            "PC 브리지 시작"
        )
        self.start_button.clicked.connect(
            self.start_bridge
        )
        button_row.addWidget(
            self.start_button
        )

        self.stop_button = QPushButton(
            "브리지 중지"
        )
        self.stop_button.clicked.connect(
            self.stop_bridge
        )
        self.stop_button.setEnabled(
            False
        )
        button_row.addWidget(
            self.stop_button
        )

        self.record_button = QPushButton(
            "선택한 WAV 녹음 시작"
        )
        self.record_button.clicked.connect(
            self.toggle_recording
        )
        self.record_button.setEnabled(
            False
        )
        button_row.addWidget(
            self.record_button
        )

        control_layout.addLayout(
            button_row
        )

        self.connection_label = QLabel(
            "PC 브리지 중지됨"
        )
        self.connection_label.setWordWrap(
            True
        )
        self.connection_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        control_layout.addWidget(
            self.connection_label
        )

        self.level_bar = QProgressBar()
        self.level_bar.setRange(
            0,
            60,
        )
        self.level_bar.setValue(
            0
        )
        self.level_bar.setFormat(
            "Input level: -∞ dBFS"
        )
        control_layout.addWidget(
            self.level_bar
        )

        self.stats_label = QLabel(
            "Buffer: - / packets: 0 / underflow: 0ms"
        )
        self.stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        control_layout.addWidget(
            self.stats_label
        )

        root.addWidget(
            control_group
        )

        log_group = QGroupBox(
            "Phone Mic 로그"
        )
        log_layout = QVBoxLayout(
            log_group
        )

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(
            True
        )
        self.log_edit.setMaximumBlockCount(
            1500
        )
        self.log_edit.setMinimumHeight(
            140
        )
        log_layout.addWidget(
            self.log_edit
        )

        root.addWidget(
            log_group,
            1,
        )

    def _camera_settings_payload(self) -> dict:
        res=self.camera_resolution_combo.currentData()
        if not isinstance(res,tuple) or len(res)!=2: res=(1280,720)
        return {"device_id":str(self.camera_device_combo.currentData() or ""),"width":int(res[0]),"height":int(res[1]),"fps":int(self.camera_fps_combo.currentData() or 15),"jpeg_quality":float(self.camera_quality_spin.value()),"mirror":self.camera_mirror_check.isChecked(),"rotation":int(self.camera_rotation_combo.currentData() or 0),"zoom":float(self.camera_zoom_spin.value()),"brightness":int(self.camera_brightness_spin.value()),"contrast":float(self.camera_contrast_spin.value()),"saturation":float(self.camera_saturation_spin.value()),"background_mode":str(self.camera_background_combo.currentData() or "none"),"background_image":str(self.camera_background_path or ""),"background_blur":int(self.camera_blur_spin.value()),"virtual_camera":self.camera_virtual_check.isChecked()}

    def _save_camera_settings(self) -> None:
        p=self._camera_settings_payload(); self.settings.setValue("s24_camera_resolution",f"{p['width']}x{p['height']}"); self.settings.setValue("s24_camera_fps",p['fps']); self.settings.setValue("s24_camera_jpeg_quality",p['jpeg_quality']); self.settings.setValue("s24_camera_mirror",p['mirror']); self.settings.setValue("s24_camera_rotation",p['rotation']); self.settings.setValue("s24_camera_zoom",p['zoom']); self.settings.setValue("s24_camera_brightness",p['brightness']); self.settings.setValue("s24_camera_contrast",p['contrast']); self.settings.setValue("s24_camera_saturation",p['saturation']); self.settings.setValue("s24_camera_background_mode",p['background_mode']); self.settings.setValue("s24_camera_background_image",p['background_image']); self.settings.setValue("s24_camera_background_blur",p['background_blur']); self.settings.setValue("s24_camera_virtual_enabled",p['virtual_camera'])

    def choose_camera_background(self) -> None:
        start=str(Path(self.camera_background_path).parent) if self.camera_background_path else str(project_root())
        path,_=QFileDialog.getOpenFileName(self,"가상 배경 이미지 선택",start,"Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if path:
            self.camera_background_path=str(Path(path).resolve()); self.camera_background_label.setText(self.camera_background_path); self._save_camera_settings()

    def start_camera_worker(self) -> None:
        self._save_camera_settings(); self.camera_controller.start_worker_async()

    def start_phone_camera(self) -> None:
        self._save_camera_settings(); self.camera_controller.start_camera_async(self._camera_settings_payload())

    def stop_phone_camera(self) -> None:
        self.camera_controller.stop_camera_async()

    def _refresh_camera_ui(self) -> None:
        s=self.camera_controller.snapshot(); devices=s.get("camera_devices",[]); sig=tuple((str(i.get("deviceId","") or ""),str(i.get("label","") or "")) for i in devices if isinstance(i,dict))
        if sig!=self._camera_device_signature:
            prev=str(self.camera_device_combo.currentData() or ""); self.camera_device_combo.blockSignals(True); self.camera_device_combo.clear()
            if not sig: self.camera_device_combo.addItem("휴대폰 카메라 목록 대기 중","")
            else:
                for n,(did,label) in enumerate(sig): self.camera_device_combo.addItem(label or f"Camera {n+1}",did)
                idx=self.camera_device_combo.findData(prev)
                if idx>=0:self.camera_device_combo.setCurrentIndex(idx)
            self.camera_device_combo.blockSignals(False); self._camera_device_signature=sig
        state=str(s.get("controller_state","stopped")); pc=bool(s.get("phone_connected",False)); run=bool(s.get("camera_running",False)); fps=float(s.get("fps",0) or 0); drop=int(s.get("dropped_frames",0) or 0); label=str(s.get("camera_label","") or ""); bg=str(s.get("background_status","") or ""); vc=bool(s.get("virtual_camera",False)); vd=str(s.get("virtual_camera_device","") or ""); ve=str(s.get("virtual_camera_error","") or ""); err=str(s.get("controller_error","") or "")
        text=f"worker={state} / phone={'connected' if pc else 'waiting'} / camera={'ON' if run else 'OFF'} / fps={fps:.1f} / dropped={drop}"+(f" / {label}" if label else "")+(f"\nBG: {bg}" if bg else "")+(f"\nVirtualCam: ON ({vd})" if vc else (f"\nVirtualCam ERROR: {ve}" if ve else ""))+(f"\nERROR: {err}" if err else ""); self.camera_live_label.setText(text)
        seq,jpeg=self.camera_controller.latest_jpeg()
        if jpeg and seq!=self._camera_preview_seq:
            image=QImage.fromData(jpeg,"JPEG")
            if not image.isNull():
                pix=QPixmap.fromImage(image); self.camera_preview.setPixmap(pix.scaled(self.camera_preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)); self._camera_preview_seq=seq

    def choose_realtime_rvc_model(
        self,
    ) -> None:
        start_dir = (
            str(
                Path(
                    self.realtime_rvc_model_path
                ).parent
            )
            if self.realtime_rvc_model_path
            else str(
                project_root()
                / "rvc_models"
            )
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Realtime RVC 모델 선택",
            start_dir,
            "RVC model (*.pth);;All files (*.*)",
        )

        if not path:
            return

        self.realtime_rvc_model_path = str(
            Path(
                path
            ).resolve()
        )
        self.realtime_rvc_model_label.setText(
            self.realtime_rvc_model_path
        )

        self.settings.setValue(
            "phone_mic_realtime_rvc_model",
            self.realtime_rvc_model_path,
        )

    def choose_realtime_rvc_index(
        self,
    ) -> None:
        start_dir = (
            str(
                Path(
                    self.realtime_rvc_model_path
                ).parent
            )
            if self.realtime_rvc_model_path
            else str(
                project_root()
                / "rvc_models"
            )
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Realtime RVC Feature Index 선택",
            start_dir,
            "RVC index (*.index);;All files (*.*)",
        )

        if not path:
            return

        self.realtime_rvc_index_path = str(
            Path(
                path
            ).resolve()
        )
        self.realtime_rvc_index_label.setText(
            self.realtime_rvc_index_path
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_index",
            self.realtime_rvc_index_path,
        )

    def clear_realtime_rvc_index(
        self,
    ) -> None:
        self.realtime_rvc_index_path = ""
        self.realtime_rvc_index_label.setText(
            "Index 미선택 - Index Rate 0으로 동작"
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_index",
            "",
        )

    def _save_realtime_rvc_settings(
        self,
    ) -> None:
        self.settings.setValue(
            "phone_mic_realtime_rvc_enabled",
            self.realtime_rvc_enable_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_rvc_ab_debug_enabled",
            self.rvc_ab_debug_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_model",
            self.realtime_rvc_model_path,
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_index",
            self.realtime_rvc_index_path,
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_pitch",
            self.realtime_rvc_pitch_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_index_rate",
            self.realtime_rvc_index_rate_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_block_ms",
            int(
                self.realtime_rvc_block_combo.currentData()
                or 200
            ),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_guard_enabled",
            self.realtime_rvc_f0_guard_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_diagnostic_enabled",
            self.realtime_rvc_f0_diagnostic_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_rmvpe_threshold",
            self.realtime_rvc_rmvpe_threshold_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_quiet_rms_db",
            self.realtime_rvc_f0_quiet_rms_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_min_voiced_ratio",
            self.realtime_rvc_f0_min_voiced_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_max_gap_ms",
            self.realtime_rvc_f0_max_gap_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_realtime_rvc_f0_min_run_ms",
            self.realtime_rvc_f0_min_run_spin.value(),
        )

    def apply_realtime_rvc_f0_guard_balanced_preset(
        self,
    ) -> None:
        self.realtime_rvc_f0_guard_check.setChecked(
            True
        )
        self.realtime_rvc_f0_diagnostic_check.setChecked(
            True
        )
        self.realtime_rvc_rmvpe_threshold_spin.setValue(
            0.05
        )
        self.realtime_rvc_f0_quiet_rms_spin.setValue(
            -48.0
        )
        self.realtime_rvc_f0_min_voiced_spin.setValue(
            0.15
        )
        self.realtime_rvc_f0_max_gap_spin.setValue(
            30
        )
        self.realtime_rvc_f0_min_run_spin.setValue(
            40
        )
        self._save_realtime_rvc_settings()

        self.runtime.log(
            "[Realtime RVC] F0 Guard 균형값: "
            "RMVPE=0.05 / quiet=-48dBFS / "
            "min voiced=0.15 / gap=30ms / min run=40ms. "
            "RVC 엔진 재시작 후 적용됩니다."
        )

    def apply_realtime_rvc_low_latency_preset(
        self,
    ) -> None:
        block_index = self.realtime_rvc_block_combo.findData(
            100
        )

        if block_index >= 0:
            self.realtime_rvc_block_combo.setCurrentIndex(
                block_index
            )

        self.jitter_spin.setValue(
            30
        )
        self._save_realtime_rvc_settings()
        self._save_settings()

        self.runtime.log(
            "[Realtime RVC] 저지연 프리셋 설정: "
            "Block=100ms / Jitter=30ms. "
            "RVC 엔진을 재시작하면 적용됩니다."
        )

    def start_realtime_rvc(
        self,
    ) -> None:
        self._save_realtime_rvc_settings()
        self._apply_runtime_settings()

        if not self.runtime.running:
            QMessageBox.information(
                self,
                "PC 브리지 필요",
                "먼저 PC 브리지를 시작하세요. "
                "CABLE Input의 실제 sample rate가 결정된 뒤 RVC 엔진을 로드합니다.",
            )
            return

        model = str(
            self.realtime_rvc_model_path
            or ""
        ).strip()

        if (
            not model
            or not Path(
                model
            ).is_file()
        ):
            QMessageBox.warning(
                self,
                "RVC 모델 없음",
                "실시간 변환에 사용할 .pth 모델을 선택하세요.",
            )
            return

        try:
            self.runtime.start_realtime_rvc_async(
                model_path=model,
                index_path=(
                    self.realtime_rvc_index_path
                    or None
                ),
                pitch=self.realtime_rvc_pitch_spin.value(),
                index_rate=self.realtime_rvc_index_rate_spin.value(),
                block_ms=int(
                    self.realtime_rvc_block_combo.currentData()
                    or 200
                ),
                f0_guard_enabled=self.realtime_rvc_f0_guard_check.isChecked(),
                f0_diagnostic_enabled=self.realtime_rvc_f0_diagnostic_check.isChecked(),
                rmvpe_threshold=self.realtime_rvc_rmvpe_threshold_spin.value(),
                f0_quiet_rms_db=self.realtime_rvc_f0_quiet_rms_spin.value(),
                f0_min_voiced_ratio=self.realtime_rvc_f0_min_voiced_spin.value(),
                f0_max_gap_ms=self.realtime_rvc_f0_max_gap_spin.value(),
                f0_min_run_ms=self.realtime_rvc_f0_min_run_spin.value(),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Realtime RVC 시작 실패",
                f"{type(exc).__name__}: {exc}",
            )

    def stop_realtime_rvc(
        self,
    ) -> None:
        self.runtime.stop_realtime_rvc()

    def _selected_output_device(
        self,
    ) -> int | None:
        data = self.output_combo.currentData()

        if data is None:
            return None

        try:
            return int(data)
        except Exception:
            return None

    def _selected_final_monitor_output_device(
        self,
    ) -> int | None:
        data = self.final_monitor_output_combo.currentData()

        if data is None:
            return None

        try:
            return int(data)
        except Exception:
            return None

    def _selected_broadcast_input_device(
        self,
    ) -> int | None:
        data = self.broadcast_input_combo.currentData()

        if data is None:
            return None

        try:
            return int(data)
        except Exception:
            return None

    def _refresh_broadcast_input_devices(self) -> None:
        previous = self.settings.value(
            "phone_mic_broadcast_input_device_name",
            "",
            type=str,
        )

        self.broadcast_input_combo.clear()
        self._broadcast_input_devices.clear()

        devices = find_nvidia_broadcast_inputs()

        if not devices:
            self.broadcast_input_combo.addItem(
                "NVIDIA Broadcast 입력 장치 없음",
                None,
            )
            return

        selected_combo = 0

        for device_index, name in devices:
            self.broadcast_input_combo.addItem(
                f"[{device_index}] {name}",
                device_index,
            )
            self._broadcast_input_devices.append(
                (
                    int(device_index),
                    str(name),
                )
            )

            if previous and previous == name:
                selected_combo = (
                    self.broadcast_input_combo.count()
                    - 1
                )

        self.broadcast_input_combo.setCurrentIndex(
            selected_combo
        )

    def refresh_output_devices(self) -> None:
        previous = self.settings.value(
            "phone_mic_output_device_name",
            "",
            type=str,
        )

        self.output_combo.clear()
        self._devices.clear()
        self.final_monitor_output_combo.clear()
        self._final_monitor_output_devices.clear()

        if sd is None:
            self.output_combo.addItem(
                "sounddevice 미설치",
                None,
            )
            self.broadcast_input_combo.clear()
            self.broadcast_input_combo.addItem(
                "sounddevice 미설치",
                None,
            )
            self.final_monitor_output_combo.addItem(
                "sounddevice 미설치",
                None,
            )
            return

        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.output_combo.addItem(
                f"장치 조회 실패: {exc}",
                None,
            )
            return

        best_index = -1
        fallback_index = -1

        final_previous = self.settings.value(
            "phone_mic_final_monitor_output_device_name_v45",
            "",
            type=str,
        )
        final_saved_index = -1
        final_default_index = -1
        final_physical_index = -1
        final_nonvirtual_index = -1

        try:
            default_output_device = int(
                sd.default.device[1]
            )
        except Exception:
            default_output_device = -1

        virtual_keywords = (
            "cable input",
            "vb-audio",
            "voicemeeter",
            "nvidia broadcast",
            "stereo mix",
        )
        physical_keywords = (
            "speaker",
            "speakers",
            "headphone",
            "headphones",
            "스피커",
            "헤드폰",
            "이어폰",
        )

        for index, info in enumerate(
            devices
        ):
            if int(
                info.get(
                    "max_output_channels",
                    0,
                )
            ) <= 0:
                continue

            name = str(
                info.get(
                    "name",
                    f"Device {index}",
                )
            )
            label = (
                f"[{index}] {name}"
            )

            self.output_combo.addItem(
                label,
                index,
            )
            combo_index = (
                self.output_combo.count()
                - 1
            )

            self.final_monitor_output_combo.addItem(
                label,
                index,
            )
            final_combo_index = (
                self.final_monitor_output_combo.count()
                - 1
            )

            self._devices.append(
                (index, name)
            )
            self._final_monitor_output_devices.append(
                (index, name)
            )

            lower = name.lower()
            is_virtual = any(
                keyword in lower
                for keyword in virtual_keywords
            )

            if final_previous and final_previous == name:
                final_saved_index = final_combo_index

            if (
                int(index) == int(default_output_device)
                and not is_virtual
            ):
                final_default_index = final_combo_index

            if (
                final_physical_index < 0
                and not is_virtual
                and any(
                    keyword in lower
                    for keyword in physical_keywords
                )
            ):
                final_physical_index = final_combo_index

            if (
                final_nonvirtual_index < 0
                and not is_virtual
            ):
                final_nonvirtual_index = final_combo_index

            if (
                "cable input" in lower
                or "vb-audio" in lower
                or "voicemeeter input" in lower
            ):
                if best_index < 0:
                    best_index = combo_index

            if previous and previous == name:
                fallback_index = combo_index

        if fallback_index >= 0:
            self.output_combo.setCurrentIndex(
                fallback_index
            )
        elif best_index >= 0:
            self.output_combo.setCurrentIndex(
                best_index
            )

        final_choice = -1

        for candidate in (
            final_saved_index,
            final_default_index,
            final_physical_index,
            final_nonvirtual_index,
        ):
            if candidate >= 0:
                final_choice = int(candidate)
                break

        if final_choice >= 0:
            self.final_monitor_output_combo.setCurrentIndex(
                final_choice
            )

        self._refresh_broadcast_input_devices()

    def refresh_adb_status(self) -> None:
        self.runtime_label.setText(
            runtime_status_text()
        )

        adb = find_adb()

        if adb is None:
            self.adb_status_label.setText(
                "ADB 없음"
            )
            self.native_mic_status_label.setText(
                "Native Mic: ADB 필요"
            )
            return

        devices = adb_devices()

        if not devices:
            self.adb_status_label.setText(
                f"ADB OK / 기기 없음 / {adb}"
            )
            self.native_mic_status_label.setText(
                "Native Mic: Galaxy USB 연결 대기"
            )
            return

        rendered = ", ".join(
            f"{serial}={state}"
            for serial, state in devices
        )
        self.adb_status_label.setText(
            "ADB: "
            + rendered
        )

        online_serial = next(
            (
                serial
                for serial, state in devices
                if state == "device"
            ),
            None,
        )

        apk = find_native_mic_apk()

        if online_serial is None:
            self.native_mic_status_label.setText(
                "Native Mic: Galaxy 연결 대기 / "
                + (
                    f"APK={apk.name}"
                    if apk is not None
                    else "APK 미빌드"
                )
            )
        else:
            installed = native_mic_installed(
                serial=online_serial
            )
            self.native_mic_status_label.setText(
                "Native Mic: "
                + (
                    "설치됨"
                    if installed
                    else "미설치"
                )
                + " / "
                + (
                    f"APK={apk.name}"
                    if apk is not None
                    else "APK 미빌드"
                )
                + " / screen-off foreground service 지원"
            )

    def _save_settings(self) -> None:
        selected = self._selected_output_device()
        selected_name = ""

        for index, name in self._devices:
            if index == selected:
                selected_name = name
                break

        self.settings.setValue(
            "phone_mic_output_device_name",
            selected_name,
        )
        self.settings.setValue(
            "phone_mic_monitor_enabled_v35",
            self.output_enabled_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_jitter_ms",
            self.jitter_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_gain_db",
            self.gain_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_highpass_enabled",
            self.highpass_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_highpass_hz",
            self.highpass_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_gate_enabled",
            self.gate_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_gate_threshold_db",
            self.gate_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_transient_suppression_enabled",
            self.transient_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_transient_reduction_db",
            self.transient_reduction_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_smart_gain_enabled",
            self.smart_gain_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_smart_gain_target_db",
            self.smart_gain_target_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_smart_gain_max_boost_db",
            self.smart_gain_max_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_limiter_enabled",
            self.limiter_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_limiter_ceiling_db",
            self.limiter_ceiling_spin.value(),
        )
        self.settings.setValue(
            "phone_mic_record_raw_copy",
            self.raw_record_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_record_clean_copy",
            self.clean_record_check.isChecked(),
        )

        broadcast_device = self._selected_broadcast_input_device()
        broadcast_name = ""

        for index, name in self._broadcast_input_devices:
            if index == broadcast_device:
                broadcast_name = name
                break

        self.settings.setValue(
            "phone_mic_broadcast_record_enabled",
            self.broadcast_record_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_broadcast_input_device_name",
            broadcast_name,
        )

        final_monitor_device = (
            self._selected_final_monitor_output_device()
        )
        final_monitor_name = ""

        for index, name in self._final_monitor_output_devices:
            if index == final_monitor_device:
                final_monitor_name = name
                break

        self.settings.setValue(
            "phone_mic_final_monitor_enabled_v45",
            self.final_monitor_check.isChecked(),
        )
        self.settings.setValue(
            "phone_mic_final_monitor_output_device_name_v45",
            final_monitor_name,
        )
        self.settings.setValue(
            "phone_mic_final_monitor_gain_db_v45",
            self.final_monitor_gain_spin.value(),
        )

        self._save_realtime_rvc_settings()
        self._save_camera_settings()

    def _apply_runtime_settings(self) -> None:
        self.runtime.configure(
            output_device=self._selected_output_device(),
            output_enabled=self.output_enabled_check.isChecked(),
            jitter_ms=self.jitter_spin.value(),
            gain_db=self.gain_spin.value(),
            highpass_enabled=self.highpass_check.isChecked(),
            highpass_hz=self.highpass_spin.value(),
            gate_enabled=self.gate_check.isChecked(),
            gate_threshold_db=self.gate_spin.value(),
            transient_suppression_enabled=self.transient_check.isChecked(),
            transient_reduction_db=self.transient_reduction_spin.value(),
            smart_gain_enabled=self.smart_gain_check.isChecked(),
            smart_gain_target_db=self.smart_gain_target_spin.value(),
            smart_gain_max_boost_db=self.smart_gain_max_spin.value(),
            limiter_enabled=self.limiter_check.isChecked(),
            limiter_ceiling_db=self.limiter_ceiling_spin.value(),
            record_raw_copy=self.raw_record_check.isChecked(),
            record_clean_copy=self.clean_record_check.isChecked(),
            broadcast_record_enabled=self.broadcast_record_check.isChecked(),
            broadcast_input_device=self._selected_broadcast_input_device(),
            final_monitor_enabled=self.final_monitor_check.isChecked(),
            final_monitor_output_device=self._selected_final_monitor_output_device(),
            final_monitor_gain_db=self.final_monitor_gain_spin.value(),
            realtime_rvc_enabled=self.realtime_rvc_enable_check.isChecked(),
            rvc_ab_debug_enabled=self.rvc_ab_debug_check.isChecked(),
        )

    def apply_s24_voice_focus_dsp_preset(
        self,
    ) -> None:
        self.gain_spin.setValue(
            0.0
        )
        self.highpass_check.setChecked(
            True
        )
        self.highpass_spin.setValue(
            80.0
        )
        self.gate_check.setChecked(
            False
        )
        self.transient_check.setChecked(
            True
        )
        self.transient_reduction_spin.setValue(
            10.0
        )
        self.smart_gain_check.setChecked(
            True
        )
        self.smart_gain_target_spin.setValue(
            -20.0
        )
        self.smart_gain_max_spin.setValue(
            12.0
        )
        self.limiter_check.setChecked(
            True
        )
        self.limiter_ceiling_spin.setValue(
            -1.0
        )

        self._save_settings()
        self._apply_runtime_settings()

        self.runtime.log(
            "[S24 Voice Focus DSP] 적용: "
            "HPF=80Hz / Gate=OFF / Click-Key=ON 10dB / "
            "SmartGain target=-20dBFS max=+12dB / Limiter=-1dBFS"
        )

    def on_monitor_toggled(
        self,
        checked: bool,
    ) -> None:
        self._save_settings()
        self._apply_runtime_settings()

        if not self.runtime.running:
            return

        try:
            self.runtime.set_monitor_enabled(
                bool(
                    checked
                )
            )
        except Exception as exc:
            self.output_enabled_check.blockSignals(
                True
            )
            self.output_enabled_check.setChecked(
                False
            )
            self.output_enabled_check.blockSignals(
                False
            )
            self.runtime.configure(
                output_device=self._selected_output_device(),
                output_enabled=False,
                jitter_ms=self.jitter_spin.value(),
                gain_db=self.gain_spin.value(),
                highpass_enabled=self.highpass_check.isChecked(),
                highpass_hz=self.highpass_spin.value(),
                gate_enabled=self.gate_check.isChecked(),
                gate_threshold_db=self.gate_spin.value(),
                transient_suppression_enabled=self.transient_check.isChecked(),
                transient_reduction_db=self.transient_reduction_spin.value(),
                smart_gain_enabled=self.smart_gain_check.isChecked(),
                smart_gain_target_db=self.smart_gain_target_spin.value(),
                smart_gain_max_boost_db=self.smart_gain_max_spin.value(),
                limiter_enabled=self.limiter_check.isChecked(),
                limiter_ceiling_db=self.limiter_ceiling_spin.value(),
                record_raw_copy=self.raw_record_check.isChecked(),
                record_clean_copy=self.clean_record_check.isChecked(),
                broadcast_record_enabled=self.broadcast_record_check.isChecked(),
                broadcast_input_device=self._selected_broadcast_input_device(),
                final_monitor_enabled=self.final_monitor_check.isChecked(),
                final_monitor_output_device=self._selected_final_monitor_output_device(),
                final_monitor_gain_db=self.final_monitor_gain_spin.value(),
                realtime_rvc_enabled=self.realtime_rvc_enable_check.isChecked(),
                rvc_ab_debug_enabled=self.rvc_ab_debug_check.isChecked(),
            )
            QMessageBox.warning(
                self,
                "모니터 출력 전환 실패",
                f"{type(exc).__name__}: {exc}",
            )

    def on_final_monitor_toggled(
        self,
        checked: bool,
    ) -> None:
        self._save_settings()
        self._apply_runtime_settings()

        if not self.runtime.running:
            return

        if checked:
            if self._selected_broadcast_input_device() is None:
                self.final_monitor_check.blockSignals(True)
                self.final_monitor_check.setChecked(False)
                self.final_monitor_check.blockSignals(False)
                self._save_settings()

                QMessageBox.warning(
                    self,
                    "최종 보이스 입력 없음",
                    "NVIDIA Broadcast 최종 마이크 입력을 선택하세요.\n"
                    "보통 Microphone (NVIDIA Broadcast) 또는 "
                    "마이크(NVIDIA Broadcast)입니다.",
                )
                return

            if self._selected_final_monitor_output_device() is None:
                self.final_monitor_check.blockSignals(True)
                self.final_monitor_check.setChecked(False)
                self.final_monitor_check.blockSignals(False)
                self._save_settings()

                QMessageBox.warning(
                    self,
                    "모니터 출력 없음",
                    "최종 보이스를 들을 스피커/헤드폰을 선택하세요.",
                )
                return

        try:
            self.runtime.set_final_monitor_enabled(
                bool(checked)
            )
        except Exception as exc:
            self.final_monitor_check.blockSignals(True)
            self.final_monitor_check.setChecked(False)
            self.final_monitor_check.blockSignals(False)
            self._save_settings()
            self._apply_runtime_settings()

            QMessageBox.warning(
                self,
                "최종 보이스 모니터 시작 실패",
                f"{type(exc).__name__}: {exc}",
            )

    def on_final_monitor_device_changed(
        self,
        *_args,
    ) -> None:
        # During initial UI construction this signal can fire before all
        # final-monitor widgets exist.
        if not hasattr(
            self,
            "final_monitor_check",
        ):
            return

        self._save_settings()
        self._apply_runtime_settings()

        if (
            not self.runtime.running
            or not self.final_monitor_check.isChecked()
        ):
            return

        try:
            self.runtime.restart_final_voice_monitor()
        except Exception as exc:
            self.runtime.log(
                "[Final Voice Monitor] 장치 전환 실패: "
                f"{type(exc).__name__}: {exc}"
            )

    def on_final_monitor_gain_changed(
        self,
        value: float,
    ) -> None:
        self.settings.setValue(
            "phone_mic_final_monitor_gain_db_v45",
            float(value),
        )
        self.runtime.set_final_monitor_gain_db(
            float(value)
        )

    def start_bridge(self) -> None:
        if self.runtime.running:
            return

        self._save_settings()
        self._apply_runtime_settings()

        selected = self._selected_output_device()

        if (
            self.output_enabled_check.isChecked()
            and selected is None
        ):
            QMessageBox.warning(
                self,
                "출력 장치 없음",
                "Windows 출력을 사용하려면 출력 장치를 선택하세요. "
                "수신/RAW 녹음만 하려면 'Windows 출력 활성화'를 끄세요.",
            )
            return

        if self.final_monitor_check.isChecked():
            if (
                self._selected_broadcast_input_device() is None
                or self._selected_final_monitor_output_device() is None
            ):
                QMessageBox.warning(
                    self,
                    "최종 보이스 모니터 장치 확인",
                    "최종 보이스 모니터가 켜져 있지만 입력 또는 출력 장치가 없습니다.\n"
                    "NVIDIA Broadcast 최종 마이크와 내 스피커/헤드폰을 선택하세요.",
                )
                return

        try:
            self.runtime.start()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Phone Mic Bridge 시작 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self.start_button.setEnabled(
            False
        )
        self.stop_button.setEnabled(
            True
        )
        self.record_button.setEnabled(
            True
        )

        self.raw_record_check.setEnabled(
            True
        )
        self.clean_record_check.setEnabled(
            True
        )
        self.broadcast_record_check.setEnabled(
            True
        )
        self.rvc_ab_debug_check.setEnabled(
            True
        )

        self.output_combo.setEnabled(
            False
        )
        # Monitor ON/OFF is intentionally live in v3.5.
        self.output_enabled_check.setEnabled(
            True
        )

        if (
            self.realtime_rvc_enable_check.isChecked()
            and self.realtime_rvc_model_path
            and Path(
                self.realtime_rvc_model_path
            ).is_file()
        ):
            QTimer.singleShot(
                100,
                self.start_realtime_rvc,
            )

    def stop_bridge(self) -> None:
        self.runtime.stop()
        self.camera_controller.stop_worker()

        self.start_button.setEnabled(
            True
        )
        self.stop_button.setEnabled(
            False
        )
        self.record_button.setEnabled(
            False
        )
        self.record_button.setText(
            "선택한 WAV 녹음 시작"
        )

        self.output_combo.setEnabled(
            True
        )
        self.output_enabled_check.setEnabled(
            True
        )

    def _run_native_action(
        self,
        kind: str,
        action,
    ) -> None:
        if self._native_action_busy:
            self.runtime.log(
                "[Native Mic] 이전 작업이 아직 진행 중입니다."
            )
            return

        self._native_action_busy = True
        self.native_mic_install_button.setEnabled(
            False
        )
        self.native_mic_launch_button.setEnabled(
            False
        )
        self.native_mic_status_label.setText(
            f"Native Mic: {kind} 진행 중..."
        )

        def worker() -> None:
            try:
                message = str(
                    action()
                )
                self.native_action_done.emit(
                    kind,
                    True,
                    message,
                )

            except Exception as exc:
                self.native_action_done.emit(
                    kind,
                    False,
                    f"{type(exc).__name__}: {exc}",
                )

        threading.Thread(
            target=worker,
            name=f"S24NativeMic-{kind}",
            daemon=True,
        ).start()

    def _on_native_action_done(
        self,
        kind: str,
        ok: bool,
        message: str,
    ) -> None:
        self._native_action_busy = False
        self.native_mic_install_button.setEnabled(
            True
        )
        self.native_mic_launch_button.setEnabled(
            True
        )
        self.refresh_adb_status()

        if ok:
            self.runtime.log(
                f"[Native Mic] {kind} 완료: {message}"
            )

            if kind == "설치":
                QMessageBox.information(
                    self,
                    "Native Mic 설치 완료",
                    message,
                )
        else:
            self.runtime.log(
                f"[Native Mic] {kind} 실패: {message}"
            )
            QMessageBox.critical(
                self,
                f"Native Mic {kind} 실패",
                message,
            )

    def install_native_mic(
        self,
    ) -> None:
        def action() -> str:
            serial, apk = install_native_mic_app()
            return (
                f"Galaxy={serial}\\n"
                f"APK={apk}\\n"
                "설치 완료"
            )

        self._run_native_action(
            "설치",
            action,
        )

    def launch_native_mic(
        self,
    ) -> None:
        if not self.runtime.running:
            QMessageBox.information(
                self,
                "PC 브리지를 먼저 시작하세요",
                "먼저 'PC 브리지 시작'을 누른 다음 Native Mic을 실행하세요.",
            )
            return

        def action() -> str:
            serial, installed_now = launch_native_mic_app(
                auto_install=True,
                auto_start=True,
            )
            return (
                f"Galaxy={serial} / "
                "tcp:8791 reverse 완료 / "
                + (
                    "APK 자동 설치 후 "
                    if installed_now
                    else ""
                )
                + "Native foreground microphone Activity 실행"
            )

        self._run_native_action(
            "실행",
            action,
        )

    def open_phone_page(self) -> None:
        if not self.runtime.running:
            QMessageBox.information(
                self,
                "PC 브리지를 먼저 시작하세요",
                "먼저 'PC 브리지 시작'을 누른 다음 USB 연결 버튼을 누르세요.",
            )
            return

        try:
            serial, url = (
                setup_adb_reverse_and_open_browser()
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Galaxy 연결 실패",
                f"{type(exc).__name__}: {exc}",
            )
            self.refresh_adb_status()
            return

        self.runtime.log(
            f"ADB reverse 완료(8790/8791/8792): {serial} -> {url}"
        )
        self.refresh_adb_status()

    def toggle_recording(self) -> None:
        snap = self.runtime.snapshot()

        if snap["recording"]:
            path = self.runtime.stop_recording()
            self.record_button.setText(
                "선택한 WAV 녹음 시작"
            )
            self.raw_record_check.setEnabled(
                True
            )
            self.clean_record_check.setEnabled(
                True
            )
            self.broadcast_record_check.setEnabled(
                True
            )
            self.rvc_ab_debug_check.setEnabled(
                True
            )

            if path is not None:
                snap_after = self.runtime.snapshot()
                saved: list[str] = []

                raw_last = str(
                    snap_after.get(
                        "last_raw_record_path",
                        "",
                    )
                    or ""
                )
                clean_last = str(
                    snap_after.get(
                        "last_clean_record_path",
                        "",
                    )
                    or ""
                )
                broadcast_last = str(
                    snap_after.get(
                        "broadcast_last_path",
                        "",
                    )
                    or ""
                )
                ab_clean_original = str(
                    snap_after.get(
                        "rvc_ab_clean_original_path",
                        "",
                    )
                    or ""
                )
                ab_clean_rvc = str(
                    snap_after.get(
                        "rvc_ab_clean_rvc_path",
                        "",
                    )
                    or ""
                )
                ab_state = str(
                    snap_after.get(
                        "rvc_ab_state",
                        "idle",
                    )
                )

                if raw_last:
                    saved.append(
                        f"RAW: {raw_last}"
                    )

                if clean_last:
                    saved.append(
                        f"CLEAN: {clean_last}"
                    )

                if broadcast_last:
                    saved.append(
                        f"BROADCAST: {broadcast_last}"
                    )

                if ab_clean_original:
                    saved.append(
                        f"CLEAN ORIGINAL: {ab_clean_original}"
                    )

                if ab_clean_rvc:
                    saved.append(
                        f"CLEAN RVC: {ab_clean_rvc}"
                    )

                if (
                    ab_clean_original
                    and ab_clean_rvc
                ):
                    saved.append(
                        "BROADCAST ORIGINAL/RVC: "
                        f"{ab_state} - 백그라운드 렌더 후 자동 저장"
                    )

                QMessageBox.information(
                    self,
                    "녹음 저장 완료",
                    "\n".join(saved)
                    if saved
                    else str(path),
                )
            return

        if not (
            self.raw_record_check.isChecked()
            or self.clean_record_check.isChecked()
            or self.broadcast_record_check.isChecked()
            or self.rvc_ab_debug_check.isChecked()
        ):
            QMessageBox.warning(
                self,
                "녹음 형식 없음",
                "RAW / CLEAN / NVIDIA Broadcast / RVC A/B 중 하나 이상을 켜세요.",
            )
            return

        if self.rvc_ab_debug_check.isChecked():
            rt = snap.get(
                "realtime_rvc",
                {},
            )

            if not (
                self.realtime_rvc_enable_check.isChecked()
                and bool(
                    rt.get(
                        "ready",
                        False,
                    )
                )
            ):
                QMessageBox.warning(
                    self,
                    "RVC A/B 준비 필요",
                    "RVC A/B 4종 비교 저장은 Realtime RVC가 READY=True여야 합니다.\n"
                    "먼저 RVC 엔진을 로드한 뒤 다시 녹음을 시작하세요.",
                )
                return

            if self._selected_output_device() is None:
                QMessageBox.warning(
                    self,
                    "A/B 출력 장치 없음",
                    "Broadcast 원본/RVC 비교를 만들려면 Windows 출력(CABLE Input)을 선택하세요.",
                )
                return

            if self._selected_broadcast_input_device() is None:
                QMessageBox.warning(
                    self,
                    "A/B NVIDIA Broadcast 입력 없음",
                    "Broadcast 비교를 만들려면 Microphone (NVIDIA Broadcast) 입력을 선택하세요.",
                )
                return

        if (
            self.broadcast_record_check.isChecked()
            and not self.rvc_ab_debug_check.isChecked()
            and self._selected_broadcast_input_device()
            is None
        ):
            QMessageBox.warning(
                self,
                "NVIDIA Broadcast 입력 없음",
                "최종 보정음 동시 녹음이 켜져 있지만 NVIDIA Broadcast 입력 장치를 찾지 못했습니다.\n\n"
                "NVIDIA Broadcast를 실행하고 '장치 새로고침'을 누른 뒤 "
                "'Microphone (NVIDIA Broadcast)' 또는 '마이크(NVIDIA Broadcast)'를 확인하세요.",
            )
            return

        try:
            path = self.runtime.start_recording()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "녹음 시작 실패",
                f"{type(exc).__name__}: {exc}",
            )
            return

        self.record_button.setText(
            "녹음 중지"
        )
        self.raw_record_check.setEnabled(
            False
        )
        self.clean_record_check.setEnabled(
            False
        )
        self.broadcast_record_check.setEnabled(
            False
        )
        self.rvc_ab_debug_check.setEnabled(
            False
        )
        self.runtime.log(
            f"recording started path={path}"
        )

    def _refresh_runtime_ui(self) -> None:
        self._apply_runtime_settings()

        for line in self.runtime.drain_logs():
            self.log_edit.appendPlainText(
                line
            )

        snap = self.runtime.snapshot()
        self._refresh_camera_ui()

        if snap["running"]:
            if snap["connected"]:
                label = (
                    snap["device_label"]
                    or "Galaxy microphone"
                )
                self.connection_label.setText(
                    "연결됨: "
                    f"{label} / "
                    f"phone {snap['client_sample_rate'] or '?'} Hz "
                    f"→ PC {snap['sample_rate']} Hz"
                )
            else:
                self.connection_label.setText(
                    "PC 브리지 실행 중 / 휴대폰에서 '마이크 시작'을 누르세요."
                )
        else:
            self.connection_label.setText(
                "PC 브리지 중지됨"
            )

        db = float(
            snap["peak_dbfs"]
        )

        if not snap["connected"]:
            value = 0
            display = "-∞"
        else:
            value = int(
                max(
                    0.0,
                    min(
                        60.0,
                        db + 60.0,
                    ),
                )
            )
            display = f"{db:.1f}"

        self.level_bar.setValue(
            value
        )
        self.level_bar.setFormat(
            f"Input level: {display} dBFS"
        )

        age = snap["last_packet_age_ms"]
        age_text = (
            f"{age:.0f}ms"
            if age is not None
            else "-"
        )

        self.stats_label.setText(
            f"Buffer {snap['buffer_ms']:.0f}ms / "
            f"packets {snap['received_packets']} / "
            f"last {age_text} / "
            f"underflow {snap['underflow_ms']:.0f}ms / "
            f"overflow {snap['overflow_ms']:.0f}ms / "
            f"click suppress {snap['transient_hits']} / "
            f"auto gain {snap['smart_gain_db']:+.1f}dB / "
            f"clean RMS {snap['smart_gain_output_rms_db']:.1f}dBFS / "
            f"Broadcast {snap['broadcast_status']} "
            f"{snap['broadcast_peak_dbfs']:.1f}dBFS / "
            f"FinalMon "
            f"{'ON' if snap['final_monitor_running'] else 'OFF'} "
            f"{snap['final_monitor_peak_dbfs']:.1f}dBFS"
        )

        rt = snap.get(
            "realtime_rvc",
            {},
        )
        rt_state = str(
            rt.get(
                "state",
                "stopped",
            )
        )
        rt_ready = bool(
            rt.get(
                "ready",
                False,
            )
        )
        rt_ms = float(
            rt.get(
                "last_roundtrip_ms",
                0.0,
            )
            or 0.0
        )
        rt_queue = float(
            rt.get(
                "queue_ms_estimate",
                0.0,
            )
            or 0.0
        )
        rt_drop = int(
            rt.get(
                "dropped_samples",
                0,
            )
            or 0
        )
        rt_error = str(
            rt.get(
                "error",
                "",
            )
            or ""
        )

        current_block_ms = int(
            self.realtime_rvc_block_combo.currentData()
            or 200
        )
        current_jitter_ms = int(
            self.jitter_spin.value()
        )
        estimated_app_delay_ms = (
            float(
                current_block_ms
            )
            + float(
                rt_ms
            )
            + float(
                current_jitter_ms
            )
            + float(
                rt_queue
            )
        )

        self.realtime_rvc_live_label.setText(
            f"State: {rt_state} / "
            f"READY={rt_ready} / "
            f"worker roundtrip={rt_ms:.1f}ms / "
            f"queue≈{rt_queue:.0f}ms / "
            f"dropped={rt_drop}"
            + (
                f"\n앱 내부 예상 지연≈{estimated_app_delay_ms:.0f}ms "
                f"(block {current_block_ms} + worker {rt_ms:.0f} + jitter {current_jitter_ms})"
                if rt_ready
                else ""
            )
            + (
                "\n※ NVIDIA Broadcast 지연은 위 예상값에 포함되지 않습니다."
                if rt_ready
                else ""
            )
            + (
                f"\nERROR: {rt_error}"
                if rt_error
                else ""
            )
        )

        ab_state = str(
            snap.get(
                "rvc_ab_state",
                "idle",
            )
        )
        ab_error = str(
            snap.get(
                "rvc_ab_error",
                "",
            )
            or ""
        )
        ab_clean_original = str(
            snap.get(
                "rvc_ab_clean_original_path",
                "",
            )
            or ""
        )
        ab_clean_rvc = str(
            snap.get(
                "rvc_ab_clean_rvc_path",
                "",
            )
            or ""
        )
        ab_broadcast_original = str(
            snap.get(
                "rvc_ab_broadcast_original_path",
                "",
            )
            or ""
        )
        ab_broadcast_rvc = str(
            snap.get(
                "rvc_ab_broadcast_rvc_path",
                "",
            )
            or ""
        )
        ab_report = str(
            snap.get(
                "rvc_ab_report_path",
                "",
            )
            or ""
        )
        ab_alignment = snap.get(
            "rvc_ab_clean_rvc_alignment",
            {},
        )

        ab_lines = [
            f"A/B: {ab_state}",
        ]

        if ab_error:
            ab_lines.append(
                f"ERROR: {ab_error}"
            )

        if ab_clean_original:
            ab_lines.append(
                f"CLEAN original: {ab_clean_original}"
            )

        if ab_clean_rvc:
            ab_lines.append(
                f"CLEAN RVC: {ab_clean_rvc}"
            )

        if ab_broadcast_original:
            ab_lines.append(
                f"Broadcast original: {ab_broadcast_original}"
            )

        if ab_broadcast_rvc:
            ab_lines.append(
                f"Broadcast RVC: {ab_broadcast_rvc}"
            )

        if (
            isinstance(
                ab_alignment,
                dict,
            )
            and ab_alignment
        ):
            ab_lines.append(
                "CLEAN RVC alignment: "
                f"{float(ab_alignment.get('lag_ms', 0.0)):.0f}ms / "
                f"corr={float(ab_alignment.get('correlation', 0.0)):.3f}"
            )

        if ab_report:
            ab_lines.append(
                f"Report: {ab_report}"
            )

        self.rvc_ab_status_label.setText(
            "\n".join(
                ab_lines
            )
        )

        if snap["recording"]:
            self.record_button.setText(
                "녹음 중지"
            )
        elif self.runtime.running:
            self.record_button.setText(
                "선택한 WAV 녹음 시작"
            )

    def shutdown(self) -> None:
        self._save_settings()
        self.runtime.stop()
        self.camera_controller.stop_worker()
