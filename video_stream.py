"""
video_stream.py
================
Grabs frames from the USB webcam on a dedicated thread so that OpenCV's
blocking `cap.read()` never stalls the inference loop -- otherwise, on a
slower device like the Orin Nano, your effective FPS is capped by capture
latency rather than by the model. Always exposes only the LATEST frame
(drops stale ones), which keeps gesture timing accurate.
"""

from __future__ import annotations
import threading
import time
import cv2


class ThreadedVideoStream:
    def __init__(self, src: int = -1, width: int = 1280, height: int = 720, fps: int = 30):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # MJPG usually gives a much higher achievable FPS than raw YUYV on
        # cheap USB webcams -- important headroom on the Orin Nano.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {src}")

        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._stopped = False

        ok, frame = self.cap.read()
        if ok:
            self._frame, self._timestamp = frame, time.time()

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
                self._timestamp = time.time()

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()
