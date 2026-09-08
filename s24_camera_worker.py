from __future__ import annotations

# V40_S24_CAMERA_VIRTUAL_WEBCAM_PATCH

import argparse, asyncio, contextlib, json, queue, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, websockets

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MEDIAPIPE_ERROR = f"{type(exc).__name__}: {exc}"
else:
    MEDIAPIPE_ERROR = ""

try:
    import pyvirtualcam
except Exception as exc:
    pyvirtualcam = None
    PYVIRTUALCAM_ERROR = f"{type(exc).__name__}: {exc}"
else:
    PYVIRTUALCAM_ERROR = ""


def _cover_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    target = width / height
    source = w / h
    if source > target:
        cw = int(round(h * target)); x0 = max(0, (w - cw)//2); crop = image[:, x0:x0+cw]
    else:
        ch = int(round(w / target)); y0 = max(0, (h - ch)//2); crop = image[y0:y0+ch, :]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


class CameraState:
    def __init__(self, video_port: int, control_port: int) -> None:
        self.video_port = int(video_port); self.control_port = int(control_port)
        self.lock = threading.RLock(); self.stop = threading.Event()
        self.ws_loop = None; self.ws_client = None
        self.phone_connected = False; self.camera_running = False; self.phone_state = "disconnected"
        self.camera_devices = []; self.phone_camera_label = ""; self.phone_camera_settings = {}
        self.frame_queue = queue.Queue(maxsize=2)
        self.received_frames = 0; self.dropped_frames = 0; self.processed_frames = 0
        self.frame_seq = 0; self.last_frame_monotonic = 0.0; self.fps = 0.0; self._fps_window = []
        self.latest_jpeg = b""
        self.settings = {
            "device_id":"", "width":1280, "height":720, "fps":15, "jpeg_quality":0.72,
            "mirror":True, "rotation":0, "zoom":1.0, "brightness":0, "contrast":1.0,
            "saturation":1.0, "background_mode":"none", "background_image":"",
            "background_blur":25, "virtual_camera":False,
        }
        self.background_image_cache_path = ""; self.background_image_cache = None
        self.segmenter = None
        self.background_status = "MediaPipe ready" if mp is not None else "MediaPipe unavailable: " + MEDIAPIPE_ERROR
        self.virtual_cam = None; self.virtual_signature = None
        self.virtual_camera_device = ""; self.virtual_camera_error = ""

    def update_settings(self, values: dict) -> None:
        v = dict(values or {})
        with self.lock:
            if "device_id" in v: self.settings["device_id"] = str(v.get("device_id","") or "")
            for key, lo, hi, default in (("width",320,3840,1280),("height",240,2160,720),("fps",5,60,15),("background_blur",3,60,25)):
                if key in v:
                    try: val = int(v.get(key, default))
                    except Exception: val = default
                    self.settings[key] = max(lo, min(val, hi))
            if "jpeg_quality" in v:
                try: val = float(v.get("jpeg_quality",.72))
                except Exception: val = .72
                self.settings["jpeg_quality"] = max(.30, min(val,.95))
            if "mirror" in v: self.settings["mirror"] = bool(v.get("mirror"))
            if "rotation" in v:
                try: val = int(v.get("rotation",0))
                except Exception: val = 0
                self.settings["rotation"] = val if val in {0,90,180,270} else 0
            if "zoom" in v:
                try: val = float(v.get("zoom",1.0))
                except Exception: val = 1.0
                self.settings["zoom"] = max(1.0,min(val,4.0))
            if "brightness" in v:
                try: val = int(v.get("brightness",0))
                except Exception: val = 0
                self.settings["brightness"] = max(-100,min(val,100))
            for key in ("contrast","saturation"):
                if key in v:
                    try: val = float(v.get(key,1.0))
                    except Exception: val = 1.0
                    self.settings[key] = max(0.0,min(val,3.0))
            if "background_mode" in v:
                val = str(v.get("background_mode","none") or "none").lower()
                self.settings["background_mode"] = val if val in {"none","blur","image"} else "none"
            if "background_image" in v: self.settings["background_image"] = str(v.get("background_image","") or "")
            if "virtual_camera" in v:
                self.settings["virtual_camera"] = bool(v.get("virtual_camera"))
                if not self.settings["virtual_camera"]: self._close_virtual_camera()

    def status(self) -> dict:
        with self.lock:
            age = (time.monotonic()-self.last_frame_monotonic)*1000 if self.last_frame_monotonic>0 else None
            return {
                "worker":"ready", "phone_connected":self.phone_connected, "camera_running":self.camera_running,
                "camera_devices":list(self.camera_devices), "camera_label":self.phone_camera_label,
                "camera_settings":dict(self.phone_camera_settings), "phone_state":self.phone_state,
                "received_frames":self.received_frames, "processed_frames":self.processed_frames,
                "dropped_frames":self.dropped_frames, "frame_seq":self.frame_seq, "fps":self.fps,
                "last_frame_age_ms":age, "settings":dict(self.settings), "background_status":self.background_status,
                "mediapipe_available":mp is not None, "virtual_camera_available":pyvirtualcam is not None,
                "virtual_camera":self.virtual_cam is not None, "virtual_camera_device":self.virtual_camera_device,
                "virtual_camera_error":self.virtual_camera_error,
            }

    def queue_frame(self, payload: bytes) -> None:
        self.received_frames += 1; self.last_frame_monotonic = time.monotonic()
        while self.frame_queue.full():
            try: self.frame_queue.get_nowait(); self.dropped_frames += 1
            except queue.Empty: break
        try: self.frame_queue.put_nowait(bytes(payload))
        except queue.Full: self.dropped_frames += 1

    def _ensure_segmenter(self):
        if self.segmenter is not None: return self.segmenter
        if mp is None:
            self.background_status = "MediaPipe unavailable: " + MEDIAPIPE_ERROR; return None
        try:
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            self.background_status = "MediaPipe Selfie Segmentation active"
            return self.segmenter
        except Exception as exc:
            self.background_status = f"MediaPipe segmentation error: {type(exc).__name__}: {exc}"; return None

    def _background_image(self, w: int, h: int, path: str):
        path = str(path or "").strip()
        if not path: return None
        if self.background_image_cache is None or self.background_image_cache_path != path:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                self.background_status = "배경 이미지 읽기 실패: " + path; return None
            self.background_image_cache = image; self.background_image_cache_path = path
        return _cover_resize(self.background_image_cache, w, h)

    def _apply_background(self, frame: np.ndarray, mode: str, image_path: str, blur_strength: int) -> np.ndarray:
        if mode == "none": return frame
        segmenter = self._ensure_segmenter()
        if segmenter is None: return frame
        result = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = np.asarray(result.segmentation_mask,dtype=np.float32)
        if mask.shape[:2] != frame.shape[:2]: mask = cv2.resize(mask,(frame.shape[1],frame.shape[0]))
        mask = cv2.GaussianBlur(mask,(0,0),sigmaX=3,sigmaY=3)
        alpha = np.clip((mask-.30)/.45,0,1)[:,:,None]
        if mode == "blur":
            sigma = max(3.0,float(blur_strength)); bg = cv2.GaussianBlur(frame,(0,0),sigmaX=sigma,sigmaY=sigma)
        else:
            bg = self._background_image(frame.shape[1],frame.shape[0],image_path)
            if bg is None: return frame
        out = frame.astype(np.float32)*alpha + bg.astype(np.float32)*(1-alpha)
        return np.clip(out,0,255).astype(np.uint8)

    def _close_virtual_camera(self) -> None:
        cam = self.virtual_cam; self.virtual_cam = None; self.virtual_signature = None; self.virtual_camera_device = ""
        if cam is not None:
            with contextlib.suppress(Exception): cam.close()

    def _ensure_virtual_camera(self, width: int, height: int, fps: int):
        if pyvirtualcam is None:
            self.virtual_camera_error = "pyvirtualcam unavailable: " + PYVIRTUALCAM_ERROR; return None
        sig = (int(width),int(height),int(fps))
        if self.virtual_cam is not None and self.virtual_signature == sig: return self.virtual_cam
        self._close_virtual_camera()
        try:
            cam = pyvirtualcam.Camera(width=width,height=height,fps=float(fps),fmt=pyvirtualcam.PixelFormat.RGB,print_fps=False)
            self.virtual_cam = cam; self.virtual_signature = sig; self.virtual_camera_device = str(cam.device); self.virtual_camera_error = ""
            print(f"[S24 Camera] Virtual camera opened: {cam.device} / {width}x{height}@{fps}",flush=True)
            return cam
        except Exception as exc:
            self.virtual_camera_error = f"{type(exc).__name__}: {exc}"; self._close_virtual_camera(); return None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self.lock: s = dict(self.settings)
        rot = int(s.get("rotation",0))
        if rot == 90: frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180: frame = cv2.rotate(frame,cv2.ROTATE_180)
        elif rot == 270: frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
        zoom = max(1.0,float(s.get("zoom",1.0)))
        if zoom > 1.001:
            h,w = frame.shape[:2]; cw=max(2,int(round(w/zoom))); ch=max(2,int(round(h/zoom))); x0=(w-cw)//2; y0=(h-ch)//2
            frame = cv2.resize(frame[y0:y0+ch,x0:x0+cw],(w,h),interpolation=cv2.INTER_LINEAR)
        if bool(s.get("mirror",True)): frame = cv2.flip(frame,1)
        contrast=float(s.get("contrast",1.0)); brightness=int(s.get("brightness",0))
        if abs(contrast-1)>1e-3 or brightness:
            frame=np.clip(frame.astype(np.float32)*contrast+brightness,0,255).astype(np.uint8)
        sat=float(s.get("saturation",1.0))
        if abs(sat-1)>1e-3:
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*sat,0,255)
            frame=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        return self._apply_background(frame,str(s.get("background_mode","none")),str(s.get("background_image","")),int(s.get("background_blur",25)))

    def frame_worker(self) -> None:
        while not self.stop.is_set():
            try: payload=self.frame_queue.get(timeout=.2)
            except queue.Empty: continue
            frame=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None: continue
            processed=self._process_frame(frame)
            with self.lock: s=dict(self.settings)
            fps=max(5,int(s.get("fps",15)))
            if bool(s.get("virtual_camera",False)):
                cam=self._ensure_virtual_camera(processed.shape[1],processed.shape[0],fps)
                if cam is not None:
                    try: cam.send(cv2.cvtColor(processed,cv2.COLOR_BGR2RGB))
                    except Exception as exc: self.virtual_camera_error=f"{type(exc).__name__}: {exc}"; self._close_virtual_camera()
            else: self._close_virtual_camera()
            ok,encoded=cv2.imencode('.jpg',processed,[cv2.IMWRITE_JPEG_QUALITY,82])
            if ok:
                now=time.monotonic()
                with self.lock:
                    self.latest_jpeg=encoded.tobytes(); self.processed_frames+=1; self.frame_seq+=1; self._fps_window.append(now)
                    self._fps_window=[x for x in self._fps_window if x>=now-2]
                    self.fps=(len(self._fps_window)-1)/(self._fps_window[-1]-self._fps_window[0]) if len(self._fps_window)>=2 and self._fps_window[-1]>self._fps_window[0] else 0.0
        self._close_virtual_camera()
        if self.segmenter is not None:
            with contextlib.suppress(Exception): self.segmenter.close()

    async def send_to_phone(self,payload:dict)->bool:
        if self.ws_client is None: return False
        try: await self.ws_client.send(json.dumps(payload,ensure_ascii=False)); return True
        except Exception: return False

    def send_to_phone_threadsafe(self,payload:dict)->bool:
        if self.ws_loop is None or not self.ws_loop.is_running() or self.ws_client is None: return False
        fut=asyncio.run_coroutine_threadsafe(self.send_to_phone(payload),self.ws_loop)
        try: return bool(fut.result(timeout=2))
        except Exception: return False

    async def video_handler(self,websocket,path=None)->None:
        self.ws_client=websocket; self.phone_connected=True; self.phone_state="connected"
        print("[S24 Camera] phone video websocket connected",flush=True)
        try:
            async for message in websocket:
                if isinstance(message,str):
                    try: p=json.loads(message)
                    except json.JSONDecodeError: continue
                    typ=str(p.get("type",""))
                    if typ=="camera_devices":
                        ds=p.get("devices",[])
                        if isinstance(ds,list): self.camera_devices=[{"deviceId":str(i.get("deviceId","") or ""),"label":str(i.get("label","") or "")} for i in ds if isinstance(i,dict)]
                    elif typ=="camera_hello":
                        self.camera_running=True; self.phone_camera_label=str(p.get("label","") or ""); st=p.get("settings",{}); self.phone_camera_settings=dict(st) if isinstance(st,dict) else {}; self.phone_state="streaming"
                    elif typ=="camera_state":
                        self.phone_state=str(p.get("state","") or ""); self.camera_running=bool(p.get("running",False))
                elif isinstance(message,(bytes,bytearray,memoryview)): self.queue_frame(bytes(message))
        except Exception as exc:
            print(f"[S24 Camera] video websocket ended: {type(exc).__name__}: {exc}",flush=True)
        finally:
            if self.ws_client is websocket: self.ws_client=None
            self.phone_connected=False; self.camera_running=False; self.phone_state="disconnected"

    async def ws_main(self)->None:
        self.ws_loop=asyncio.get_running_loop()
        async with websockets.serve(self.video_handler,"127.0.0.1",self.video_port,max_size=16*1024*1024,ping_interval=10,ping_timeout=20):
            print(f"[S24 Camera] video websocket: ws://127.0.0.1:{self.video_port}/",flush=True)
            while not self.stop.is_set(): await asyncio.sleep(.2)


class ControlHandler(BaseHTTPRequestHandler):
    state: CameraState|None=None
    def _json(self,payload,status=200):
        body=json.dumps(payload,ensure_ascii=False).encode('utf-8'); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        s=self.state
        if s is None: return self._json({'error':'state unavailable'},500)
        if self.path=='/status': return self._json(s.status())
        if self.path=='/frame.jpg':
            with s.lock: body=bytes(s.latest_jpeg)
            if not body: self.send_response(204); self.end_headers(); return
            self.send_response(200); self.send_header('Content-Type','image/jpeg'); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body); return
        self._json({'error':'not found'},404)
    def do_POST(self):
        s=self.state
        if s is None: return self._json({'error':'state unavailable'},500)
        try:
            n=int(self.headers.get('Content-Length','0') or 0); p=json.loads(self.rfile.read(max(0,n)).decode('utf-8') if n else '{}')
        except Exception as exc: return self._json({'error':f'{type(exc).__name__}: {exc}'},400)
        if self.path!='/control': return self._json({'error':'not found'},404)
        action=str(p.get('action','update') or 'update').lower(); settings=p.get('settings',{})
        if isinstance(settings,dict): s.update_settings(settings)
        if action=='shutdown':
            s.stop.set(); s.send_to_phone_threadsafe({'type':'camera_control','action':'stop'}); return self._json({'ok':True})
        if action=='stop': return self._json({'ok':True,'phone_command_sent':s.send_to_phone_threadsafe({'type':'camera_control','action':'stop'})})
        if action=='start':
            with s.lock: c=dict(s.settings)
            cmd={'type':'camera_control','action':'start','deviceId':c.get('device_id',''),'width':int(c.get('width',1280)),'height':int(c.get('height',720)),'fps':int(c.get('fps',15)),'quality':float(c.get('jpeg_quality',.72))}
            return self._json({'ok':True,'phone_command_sent':s.send_to_phone_threadsafe(cmd)})
        self._json({'ok':True})
    def log_message(self,format,*args): return


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--video-port',type=int,default=8792); p.add_argument('--control-port',type=int,default=8793); a=p.parse_args()
    state=CameraState(a.video_port,a.control_port)
    ft=threading.Thread(target=state.frame_worker,daemon=True); ft.start()
    handler=type('S24CameraControlHandler',(ControlHandler,),{'state':state})
    http=ThreadingHTTPServer(('127.0.0.1',a.control_port),handler); ht=threading.Thread(target=http.serve_forever,daemon=True); ht.start()
    print(f"VPA_S24_CAMERA_READY|video_port={a.video_port}|control_port={a.control_port}|mediapipe={'yes' if mp is not None else 'no'}|pyvirtualcam={'yes' if pyvirtualcam is not None else 'no'}",flush=True)
    try: asyncio.run(state.ws_main())
    except KeyboardInterrupt: pass
    finally:
        state.stop.set(); state._close_virtual_camera()
        with contextlib.suppress(Exception): http.shutdown(); http.server_close()
        if ft.is_alive(): ft.join(timeout=2)
    return 0

if __name__=='__main__': raise SystemExit(main())
