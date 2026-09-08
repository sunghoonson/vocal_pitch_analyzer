from __future__ import annotations

# V40_S24_CAMERA_VIRTUAL_WEBCAM_PATCH

from pathlib import Path
from typing import Callable
import json
import os
import subprocess
import threading
import time
import urllib.request


LogCallback = Callable[[str], None]


def project_root() -> Path:
    return Path(__file__).resolve().parent


def camera_python() -> Path:
    if os.name == "nt":
        return project_root() / ".venv_camera" / "Scripts" / "python.exe"
    return project_root() / ".venv_camera" / "bin" / "python"


def camera_worker_path() -> Path:
    return project_root() / "s24_camera_worker.py"


def camera_runtime_status_text() -> str:
    missing = []
    if not camera_python().is_file():
        missing.append(".venv_camera")
    if not camera_worker_path().is_file():
        missing.append("s24_camera_worker.py")
    if missing:
        return "S24 Camera runtime 준비 안 됨: " + ", ".join(missing) + " / SETUP_S24_CAMERA_BRIDGE.bat 실행"
    return "S24 Camera runtime OK / .venv_camera (OpenCV + MediaPipe + pyvirtualcam)"


class S24CameraController:
    def __init__(self, *, video_port: int = 8792, control_port: int = 8793, log_callback: LogCallback | None = None) -> None:
        self.video_port = int(video_port)
        self.control_port = int(control_port)
        self.log_callback = log_callback
        self.process: subprocess.Popen | None = None
        self.state = "stopped"
        self.error = ""
        self._lock = threading.Lock()
        self._status = {
            "worker": "stopped",
            "phone_connected": False,
            "camera_running": False,
            "camera_devices": [],
            "frame_seq": 0,
            "fps": 0.0,
            "virtual_camera": False,
            "virtual_camera_device": "",
            "virtual_camera_error": "",
            "background_status": "",
        }
        self._latest_jpeg = b""
        self._latest_frame_seq = -1
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._operation_lock = threading.Lock()

    def _log(self, text: str) -> None:
        value = str(text).strip()
        if value and self.log_callback is not None:
            self.log_callback("[S24 Camera] " + value)

    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self.control_port}"

    def _get(self, path: str, *, timeout: float = 1.0) -> bytes:
        req = urllib.request.Request(self._base_url() + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read()

    def _post(self, path: str, payload: dict, *, timeout: float = 2.0) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._base_url() + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = res.read()
        return json.loads(data.decode("utf-8")) if data else {}

    def _pipe_reader(self, pipe, prefix: str) -> None:
        if pipe is None:
            return
        try:
            for raw in iter(pipe.readline, ""):
                line = str(raw).strip()
                if line:
                    self._log(prefix + line)
        except Exception:
            pass

    def _launch_worker(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        py = camera_python()
        worker = camera_worker_path()
        if not py.is_file():
            raise RuntimeError("카메라 전용 가상환경이 없습니다. SETUP_S24_CAMERA_BRIDGE.bat을 먼저 실행하세요.")
        if not worker.is_file():
            raise RuntimeError(f"Camera worker not found: {worker}")
        command = [str(py), "-u", str(worker), "--video-port", str(self.video_port), "--control-port", str(self.control_port)]
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.state = "starting"
        self.error = ""
        self.process = subprocess.Popen(
            command,
            cwd=str(project_root()),
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
        threading.Thread(target=self._pipe_reader, args=(self.process.stdout, ""), daemon=True).start()
        threading.Thread(target=self._pipe_reader, args=(self.process.stderr, "stderr: "), daemon=True).start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"S24 Camera worker가 시작 중 종료되었습니다. exit={self.process.returncode}")
            try:
                status = json.loads(self._get("/status", timeout=0.5).decode("utf-8"))
                with self._lock:
                    self._status = status
                self.state = "ready"
                self.error = ""
                self._log(f"worker READY / video ws={self.video_port} / control={self.control_port}")
                self._ensure_poll_thread()
                return
            except Exception:
                time.sleep(0.15)
        raise TimeoutError("S24 Camera worker 시작 시간 초과")

    def _ensure_poll_thread(self) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="S24CameraPoll", daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            process = self.process
            if process is None or process.poll() is not None:
                if self.state not in {"stopped", "error"}:
                    self.state = "error"
                    self.error = "camera worker exited"
                break
            try:
                status = json.loads(self._get("/status", timeout=0.7).decode("utf-8"))
                frame_seq = int(status.get("frame_seq", 0) or 0)
                if frame_seq > 0 and frame_seq != self._latest_frame_seq:
                    try:
                        jpeg = self._get("/frame.jpg", timeout=0.7)
                        if jpeg:
                            with self._lock:
                                self._latest_jpeg = bytes(jpeg)
                                self._latest_frame_seq = frame_seq
                    except Exception:
                        pass
                with self._lock:
                    self._status = status
                self.state = "ready"
            except Exception as exc:
                if not self._stop.is_set():
                    self.error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(0.15)

    def _operation(self, func) -> None:
        def runner() -> None:
            with self._operation_lock:
                try:
                    func()
                except Exception as exc:
                    self.state = "error"
                    self.error = f"{type(exc).__name__}: {exc}"
                    self._log("ERROR: " + self.error)
        threading.Thread(target=runner, name="S24CameraOperation", daemon=True).start()

    def start_worker_async(self) -> None:
        self._operation(self._launch_worker)

    def start_camera_async(self, settings: dict) -> None:
        def operation() -> None:
            self._launch_worker()
            self._post("/control", {"action": "start", "settings": settings}, timeout=3.0)
            self._log("휴대폰 카메라 시작/재시작 요청")
        self._operation(operation)

    def apply_processing_async(self, settings: dict) -> None:
        def operation() -> None:
            self._launch_worker()
            self._post("/control", {"action": "update", "settings": settings}, timeout=3.0)
        self._operation(operation)

    def stop_camera_async(self) -> None:
        def operation() -> None:
            if self.state == "ready":
                self._post("/control", {"action": "stop"}, timeout=2.0)
                self._log("휴대폰 카메라 중지 요청")
        self._operation(operation)

    def snapshot(self) -> dict:
        with self._lock:
            status = dict(self._status)
        status["controller_state"] = str(self.state)
        status["controller_error"] = str(self.error)
        return status

    def latest_jpeg(self) -> tuple[int, bytes]:
        with self._lock:
            return int(self._latest_frame_seq), bytes(self._latest_jpeg)

    def stop_worker(self) -> None:
        self._stop.set()
        try:
            if self.state == "ready":
                self._post("/control", {"action": "shutdown"}, timeout=1.5)
        except Exception:
            pass
        process = self.process
        self.process = None
        if process is not None:
            try:
                process.wait(timeout=3.0)
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
        self._poll_thread = None
        self.state = "stopped"
        self.error = ""
        with self._lock:
            self._status = {
                "worker": "stopped", "phone_connected": False, "camera_running": False,
                "camera_devices": [], "frame_seq": 0, "fps": 0.0,
                "virtual_camera": False, "virtual_camera_device": "",
                "virtual_camera_error": "", "background_status": "",
            }
            self._latest_jpeg = b""
            self._latest_frame_seq = -1
