from __future__ import annotations

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

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QDesktopServices
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
    QMessageBox,
    QPlainTextEdit,
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

try:
    import websockets
except Exception as exc:  # pragma: no cover - depends on runtime installation
    websockets = None
    _WEBSOCKETS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _WEBSOCKETS_IMPORT_ERROR = ""


HTTP_PORT = 8790
WS_PORT = 8791
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


def setup_adb_reverse_and_open_browser(
    *,
    http_port: int = HTTP_PORT,
    ws_port: int = WS_PORT,
) -> tuple[str, str]:
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

    if not online:
        if unauthorized:
            raise RuntimeError(
                "Galaxy가 USB로 보이지만 아직 인증되지 않았습니다. "
                "휴대폰에 나타난 'USB 디버깅을 허용할까요?' 창에서 허용한 뒤 다시 시도하세요."
            )

        raise RuntimeError(
            "ADB에서 연결된 Android 기기를 찾지 못했습니다. "
            "S24 Ultra의 개발자 옵션 > USB 디버깅을 켜고 USB 케이블로 연결하세요."
        )

    serial = online[0]

    for port in (int(http_port), int(ws_port)):
        result = _run_adb(
            [
                "reverse",
                f"tcp:{port}",
                f"tcp:{port}",
            ],
            serial=serial,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"adb reverse tcp:{port} 실패:\n"
                + (result.stderr or result.stdout or "(출력 없음)")
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
) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>S24 Ultra Phone Mic Bridge</title>
<style>
:root {{ color-scheme: dark; }}
body {{
  margin: 0;
  padding: 20px;
  font-family: system-ui, -apple-system, "Noto Sans KR", sans-serif;
  background: #111318;
  color: #f2f3f5;
}}
.card {{
  max-width: 760px;
  margin: 0 auto 14px auto;
  padding: 18px;
  border-radius: 16px;
  background: #1d2128;
}}
h1 {{ font-size: 24px; margin: 0 0 12px 0; }}
h2 {{ font-size: 17px; margin: 0 0 10px 0; }}
p {{ line-height: 1.55; color: #cfd3da; }}
button, select {{
  width: 100%;
  min-height: 52px;
  margin-top: 10px;
  border: 0;
  border-radius: 12px;
  font-size: 16px;
  padding: 10px 12px;
}}
button {{ font-weight: 700; }}
#start {{ background: #4d8dff; color: white; }}
#stop {{ background: #3a3e46; color: white; }}
#meter {{
  width: 100%;
  height: 18px;
  background: #30343b;
  border-radius: 10px;
  overflow: hidden;
}}
#bar {{
  width: 0%;
  height: 100%;
  background: #6ea1ff;
  transition: width 80ms linear;
}}
.small {{
  font-size: 13px;
  color: #9fa7b2;
  word-break: break-all;
}}
.ok {{ color: #76d39b; }}
.warn {{ color: #ffcc66; }}
.bad {{ color: #ff7f7f; }}
.option-row {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 10px;
  color: #d8dce3;
}}
.option-row input {{
  width: 22px;
  height: 22px;
}}
</style>
</head>
<body>
<div class="card">
  <h1>Galaxy S24 Ultra → PC 마이크</h1>
  <p>
    휴대폰 마이크를 20ms PCM 패킷으로 PC에 전송합니다.
    USB/ADB reverse를 사용하므로 이 페이지는 휴대폰의 localhost에서 열립니다.
  </p>
  <div id="secure" class="small"></div>
</div>

<div class="card">
  <h2>마이크</h2>
  <select id="device"></select>
  <label class="option-row">
    <input id="noiseSuppression" type="checkbox" checked>
    <span>브라우저 Noise Suppression 사용 (키보드/마우스/생활소음 억제, 권장)</span>
  </label>
  <label class="option-row">
    <input id="echoCancellation" type="checkbox">
    <span>Echo Cancellation 사용 (PC 스피커 모니터를 켤 때만 권장)</span>
  </label>
  <label class="option-row">
    <input id="autoGainControl" type="checkbox">
    <span>Auto Gain Control 사용 (학습용 녹음에는 보통 OFF 권장)</span>
  </label>
  <button id="start">마이크 시작</button>
  <button id="stop" disabled>중지</button>
  <p id="state" class="small">대기 중</p>
  <div id="meter"><div id="bar"></div></div>
</div>

<div class="card">
  <h2>권장</h2>
  <p class="small">
    마이크 권한을 허용하세요. 녹음 중에는 화면을 켜 두는 것이 가장 안정적입니다.
    Galaxy Buds가 연결되어 있다면 마이크 목록에서 휴대폰 내장 마이크가 아닌
    Bluetooth 입력이 선택될 수 있으므로 장치 이름을 확인하세요.
  </p>
</div>

<script>
const WS_PORT = {int(ws_port)};
const PACKET_FRAMES = 960; // 20ms at 48 kHz.

let ws = null;
let stream = null;
let context = null;
let source = null;
let worklet = null;
let sink = null;
let wakeLock = null;
let meterTimer = null;
let analyser = null;

const deviceSelect = document.getElementById("device");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const state = document.getElementById("state");
const bar = document.getElementById("bar");
const secure = document.getElementById("secure");
const noiseSuppressionCheck = document.getElementById("noiseSuppression");
const echoCancellationCheck = document.getElementById("echoCancellation");
const autoGainControlCheck = document.getElementById("autoGainControl");

secure.textContent =
  "Secure context: " + window.isSecureContext +
  " / page=" + location.href;

secure.className = window.isSecureContext ? "small ok" : "small bad";

function setState(text, cls="small") {{
  state.textContent = text;
  state.className = cls;
}}

async function refreshDevices() {{
  let devices = [];
  try {{
    devices = await navigator.mediaDevices.enumerateDevices();
  }} catch (err) {{
    setState("장치 목록 실패: " + err, "small warn");
    return;
  }}

  const inputs = devices.filter(d => d.kind === "audioinput");
  const previous = deviceSelect.value;
  deviceSelect.innerHTML = "";

  inputs.forEach((d, i) => {{
    const option = document.createElement("option");
    option.value = d.deviceId;
    option.textContent = d.label || ("마이크 " + (i + 1));
    deviceSelect.appendChild(option);
  }});

  if ([...deviceSelect.options].some(o => o.value === previous)) {{
    deviceSelect.value = previous;
  }}
}}

function connectSocket() {{
  return new Promise((resolve, reject) => {{
    const url = "ws://localhost:" + WS_PORT + "/";
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";

    const timeout = setTimeout(() => {{
      try {{ socket.close(); }} catch (_) {{}}
      reject(new Error("WebSocket 연결 시간 초과"));
    }}, 5000);

    socket.onopen = () => {{
      clearTimeout(timeout);
      ws = socket;
      resolve(socket);
    }};

    socket.onerror = () => {{
      clearTimeout(timeout);
      reject(new Error("PC WebSocket 연결 실패"));
    }};

    socket.onclose = () => {{
      if (ws === socket) {{
        ws = null;
      }}
    }};
  }});
}}

async function requestWakeLock() {{
  try {{
    if ("wakeLock" in navigator) {{
      wakeLock = await navigator.wakeLock.request("screen");
    }}
  }} catch (_) {{}}
}}

async function startMic() {{
  if (!window.isSecureContext || !navigator.mediaDevices) {{
    throw new Error(
      "브라우저가 마이크 API를 허용하지 않습니다. " +
      "PC 버튼으로 adb reverse를 설정한 뒤 http://localhost 페이지를 여세요."
    );
  }}

  if (!ws || ws.readyState !== WebSocket.OPEN) {{
    await connectSocket();
  }}

  const selected = deviceSelect.value;
  const constraints = {{
    audio: {{
      deviceId: selected ? {{ exact: selected }} : undefined,
      channelCount: {{ ideal: 1 }},
      sampleRate: {{ ideal: 48000 }},
      echoCancellation: echoCancellationCheck.checked,
      noiseSuppression: noiseSuppressionCheck.checked,
      autoGainControl: autoGainControlCheck.checked
    }},
    video: false
  }};

  stream = await navigator.mediaDevices.getUserMedia(constraints);
  await refreshDevices();

  context = new AudioContext({{
    sampleRate: 48000,
    latencyHint: "interactive"
  }});

  const processorCode = `
class PhoneMicProcessor extends AudioWorkletProcessor {{
  constructor() {{
    super();
    this.packet = new Float32Array(${{PACKET_FRAMES}});
    this.offset = 0;
  }}

  process(inputs) {{
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;

    const channel = input[0];
    let pos = 0;

    while (pos < channel.length) {{
      const take = Math.min(
        channel.length - pos,
        this.packet.length - this.offset
      );

      this.packet.set(
        channel.subarray(pos, pos + take),
        this.offset
      );

      this.offset += take;
      pos += take;

      if (this.offset >= this.packet.length) {{
        const ready = this.packet;
        this.packet = new Float32Array(${{PACKET_FRAMES}});
        this.offset = 0;
        this.port.postMessage(ready.buffer, [ready.buffer]);
      }}
    }}

    return true;
  }}
}}
registerProcessor("phone-mic-processor", PhoneMicProcessor);
`;

  const blob = new Blob(
    [processorCode],
    {{ type: "application/javascript" }}
  );
  const moduleURL = URL.createObjectURL(blob);
  await context.audioWorklet.addModule(moduleURL);
  URL.revokeObjectURL(moduleURL);

  source = context.createMediaStreamSource(stream);
  worklet = new AudioWorkletNode(
    context,
    "phone-mic-processor",
    {{
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1]
    }}
  );

  analyser = context.createAnalyser();
  analyser.fftSize = 1024;

  sink = context.createGain();
  sink.gain.value = 0.0;

  source.connect(analyser);
  source.connect(worklet);
  worklet.connect(sink);
  sink.connect(context.destination);

  worklet.port.onmessage = event => {{
    if (ws && ws.readyState === WebSocket.OPEN) {{
      ws.send(event.data);
    }}
  }};

  const track = stream.getAudioTracks()[0];
  const settings = track ? track.getSettings() : {{}};

  ws.send(JSON.stringify({{
    type: "hello",
    sampleRate: context.sampleRate,
    packetFrames: PACKET_FRAMES,
    deviceLabel: track ? track.label : "",
    trackSettings: settings,
    requestedNoiseSuppression: noiseSuppressionCheck.checked,
    requestedEchoCancellation: echoCancellationCheck.checked,
    requestedAutoGainControl: autoGainControlCheck.checked
  }}));

  const meterData = new Float32Array(analyser.fftSize);

  meterTimer = setInterval(() => {{
    if (!analyser) return;

    analyser.getFloatTimeDomainData(meterData);
    let sum = 0;

    for (let i = 0; i < meterData.length; i++) {{
      sum += meterData[i] * meterData[i];
    }}

    const rms = Math.sqrt(sum / meterData.length);
    const db = 20 * Math.log10(Math.max(rms, 1e-6));
    const pct = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
    bar.style.width = pct + "%";
  }}, 80);

  await context.resume();
  await requestWakeLock();

  startButton.disabled = true;
  stopButton.disabled = false;
  noiseSuppressionCheck.disabled = true;
  echoCancellationCheck.disabled = true;
  autoGainControlCheck.disabled = true;

  setState(
    "전송 중 / " +
    (track ? track.label : "microphone") +
    " / AudioContext " + context.sampleRate + " Hz",
    "small ok"
  );
}}

async function stopMic() {{
  if (meterTimer) {{
    clearInterval(meterTimer);
    meterTimer = null;
  }}

  if (worklet) {{
    try {{ worklet.disconnect(); }} catch (_) {{}}
    worklet = null;
  }}

  if (source) {{
    try {{ source.disconnect(); }} catch (_) {{}}
    source = null;
  }}

  if (sink) {{
    try {{ sink.disconnect(); }} catch (_) {{}}
    sink = null;
  }}

  analyser = null;

  if (stream) {{
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }}

  if (context) {{
    try {{ await context.close(); }} catch (_) {{}}
    context = null;
  }}

  if (ws) {{
    try {{ ws.close(); }} catch (_) {{}}
    ws = null;
  }}

  if (wakeLock) {{
    try {{ await wakeLock.release(); }} catch (_) {{}}
    wakeLock = null;
  }}

  bar.style.width = "0%";
  startButton.disabled = false;
  stopButton.disabled = true;
  noiseSuppressionCheck.disabled = false;
  echoCancellationCheck.disabled = false;
  autoGainControlCheck.disabled = false;
  setState("중지됨");
}}

startButton.addEventListener("click", async () => {{
  startButton.disabled = true;
  setState("마이크 권한/연결 준비 중...");

  try {{
    await startMic();
  }} catch (err) {{
    console.error(err);
    setState("시작 실패: " + err, "small bad");
    startButton.disabled = false;
    stopButton.disabled = true;
  }}
}});

stopButton.addEventListener("click", stopMic);

deviceSelect.addEventListener("change", async () => {{
  if (stream) {{
    await stopMic();
    setState("마이크 장치가 변경되었습니다. 다시 시작하세요.", "small warn");
  }}
}});

document.addEventListener("visibilitychange", async () => {{
  if (
    document.visibilityState === "visible"
    && stream
    && (!wakeLock || wakeLock.released)
  ) {{
    await requestWakeLock();
  }}
}});

if (navigator.mediaDevices) {{
  refreshDevices();
}}
</script>
</body>
</html>
"""


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
            "microphone=(self)",
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
        self.record_clean_copy = True

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
        self._clean_record_wave: wave.Wave_write | None = None
        self._clean_record_path: Path | None = None

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
        record_clean_copy: bool,
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
            self.record_clean_copy = bool(
                record_clean_copy
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

    def _start_http(self) -> None:
        page = _phone_page_html(
            ws_port=self.ws_port,
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

                        with self._stats_lock:
                            self.client_sample_rate = rate
                            self.packet_frames = packet_frames
                            self.client_device_label = label

                        self.log(
                            "Galaxy audio: "
                            f"{label or 'microphone'} / "
                            f"{rate} Hz / "
                            f"{packet_frames} frames / "
                            f"NS={'ON' if requested_ns else 'OFF'} / "
                            f"AEC={'ON' if requested_aec else 'OFF'} / "
                            f"AGC={'ON' if requested_agc else 'OFF'}"
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

                if output_enabled:
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
            if self._record_wave is not None:
                assert self._record_path is not None
                return self._record_path

            recordings_dir().mkdir(
                parents=True,
                exist_ok=True,
            )

            if path is None:
                stamp = _dt.datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                path = (
                    recordings_dir()
                    / f"s24_raw_{stamp}.wav"
                )
            else:
                path = Path(
                    path
                ).expanduser().resolve()
                stem = path.stem

                if not stem.startswith(
                    "s24_raw_"
                ):
                    stamp = _dt.datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                else:
                    stamp = stem[
                        len(
                            "s24_raw_"
                        ):
                    ]

            path = Path(
                path
            ).expanduser().resolve()
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            writer = wave.open(
                str(path),
                "wb",
            )
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(
                int(self.sample_rate)
            )

            self._record_wave = writer
            self._record_path = path

            clean_path = (
                path.parent
                / f"s24_clean_{stamp}.wav"
            )

            if self.record_clean_copy:
                clean_writer = wave.open(
                    str(clean_path),
                    "wb",
                )
                clean_writer.setnchannels(1)
                clean_writer.setsampwidth(2)
                clean_writer.setframerate(
                    int(self.sample_rate)
                )
                self._clean_record_wave = clean_writer
                self._clean_record_path = clean_path
            else:
                self._clean_record_wave = None
                self._clean_record_path = None

        self.log(
            f"RAW WAV 녹음 시작: {path}"
        )

        if self._clean_record_path is not None:
            self.log(
                "CLEAN WAV 동시 녹음 시작: "
                f"{self._clean_record_path}"
            )

        return path

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

    def stop_recording(self) -> Path | None:
        with self._record_lock:
            raw_writer = self._record_wave
            raw_path = self._record_path
            clean_writer = self._clean_record_wave
            clean_path = self._clean_record_path

            self._record_wave = None
            self._record_path = None
            self._clean_record_wave = None
            self._clean_record_path = None

            if raw_writer is not None:
                with contextlib.suppress(Exception):
                    raw_writer.close()

            if clean_writer is not None:
                with contextlib.suppress(Exception):
                    clean_writer.close()

        if raw_path is not None:
            self.log(
                f"RAW WAV 녹음 완료: {raw_path}"
            )

        if clean_path is not None:
            self.log(
                f"CLEAN WAV 녹음 완료: {clean_path}"
            )

        return raw_path

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
                self._record_wave
                is not None
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
        }


class PhoneMicBridgeWidget(QWidget):
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
        self._devices: list[tuple[int, str]] = []

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
        root = QVBoxLayout(
            self
        )

        intro = QGroupBox(
            "Galaxy S24 Ultra Phone Mic Bridge v3.6"
        )
        intro_layout = QVBoxLayout(
            intro
        )

        text = QLabel(
            "USB 케이블 + ADB reverse를 이용해 S24 Ultra의 Chrome에서 "
            "48kHz mono PCM을 PC로 전송합니다. "
            "Windows 앱에서 마이크처럼 쓰려면 출력 장치로 "
            "VB-CABLE의 'CABLE Input' 같은 가상 오디오 입력을 선택하세요."
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
            "USB 연결 후 휴대폰에 뜨는 PC 인증 창을 허용하세요. "
            "그 다음 아래 버튼을 누르면 8790/8791 포트를 reverse하고 "
            "휴대폰 Chrome에서 http://localhost:8790 을 엽니다."
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
            "USB 연결 + 휴대폰 Chrome 열기"
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
        root.addWidget(
            adb_group
        )

        output_group = QGroupBox(
            "2. Windows 오디오 출력"
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
            "실시간 스피커/Windows 모니터 출력"
        )
        self.output_enabled_check.setChecked(
            self.settings.value(
                "phone_mic_monitor_enabled_v35",
                False,
                type=bool,
            )
        )
        self.output_enabled_check.setToolTip(
            "OFF여도 휴대폰 수신과 WAV 녹음은 계속됩니다. "
            "ON이면 선택한 Windows 출력 장치로 실시간 재생합니다. "
            "일반 스피커에서는 하울링이 날 수 있으므로 평소에는 OFF를 권장합니다. "
            "Discord/OBS용 VB-CABLE을 사용할 때는 ON으로 설정하세요."
        )
        self.output_enabled_check.toggled.connect(
            self.on_monitor_toggled
        )
        output_layout.addRow(
            "",
            self.output_enabled_check,
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

        self.clean_record_check = QCheckBox(
            "CLEAN WAV 동시 저장 (PC DSP + Smart Gain 적용)"
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

        control_group = QGroupBox(
            "4. 브리지 / 녹음"
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
            "RAW + CLEAN WAV 녹음 시작"
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
        log_layout.addWidget(
            self.log_edit
        )

        root.addWidget(
            log_group,
            1,
        )

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

    def refresh_output_devices(self) -> None:
        previous = self.settings.value(
            "phone_mic_output_device_name",
            "",
            type=str,
        )

        self.output_combo.clear()
        self._devices.clear()

        if sd is None:
            self.output_combo.addItem(
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

            self._devices.append(
                (index, name)
            )

            lower = name.lower()

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

    def refresh_adb_status(self) -> None:
        self.runtime_label.setText(
            runtime_status_text()
        )

        adb = find_adb()

        if adb is None:
            self.adb_status_label.setText(
                "ADB 없음"
            )
            return

        devices = adb_devices()

        if not devices:
            self.adb_status_label.setText(
                f"ADB OK / 기기 없음 / {adb}"
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
            "phone_mic_record_clean_copy",
            self.clean_record_check.isChecked(),
        )

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
            record_clean_copy=self.clean_record_check.isChecked(),
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
                record_clean_copy=self.clean_record_check.isChecked(),
            )
            QMessageBox.warning(
                self,
                "모니터 출력 전환 실패",
                f"{type(exc).__name__}: {exc}",
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

        self.output_combo.setEnabled(
            False
        )
        # Monitor ON/OFF is intentionally live in v3.5.
        self.output_enabled_check.setEnabled(
            True
        )

    def stop_bridge(self) -> None:
        self.runtime.stop()

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
            "RAW + CLEAN WAV 녹음 시작"
        )

        self.output_combo.setEnabled(
            True
        )
        self.output_enabled_check.setEnabled(
            True
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
            f"ADB reverse 완료: {serial} -> {url}"
        )
        self.refresh_adb_status()

    def toggle_recording(self) -> None:
        snap = self.runtime.snapshot()

        if snap["recording"]:
            path = self.runtime.stop_recording()
            self.record_button.setText(
                "RAW + CLEAN WAV 녹음 시작"
            )

            if path is not None:
                QMessageBox.information(
                    self,
                    "녹음 저장 완료",
                    str(path),
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
            "RAW + CLEAN WAV 녹음 중지"
        )
        self.runtime.log(
            f"학습용 RAW recording path={path}"
        )

    def _refresh_runtime_ui(self) -> None:
        self._apply_runtime_settings()

        for line in self.runtime.drain_logs():
            self.log_edit.appendPlainText(
                line
            )

        snap = self.runtime.snapshot()

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
            f"clean RMS {snap['smart_gain_output_rms_db']:.1f}dBFS"
        )

        if snap["recording"]:
            self.record_button.setText(
                "RAW + CLEAN WAV 녹음 중지"
            )
        elif self.runtime.running:
            self.record_button.setText(
                "RAW + CLEAN WAV 녹음 시작"
            )

    def shutdown(self) -> None:
        self._save_settings()
        self.runtime.stop()
