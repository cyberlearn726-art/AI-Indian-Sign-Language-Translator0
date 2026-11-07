"""
Thread-Safe Direct Camera Capture Module using OpenCV
Direct single-thread acquisition designed specifically for worker threads.
Eliminates DirectShow cross-apartment COM deadlocks and blank feed issues.
"""

import time
import cv2
from typing import Optional, Tuple
import numpy as np

import config


class CameraStream:
    """
    Direct, low-latency camera capture manager.
    Designed to run seamlessly inside worker threads (such as QThread)
    without cross-thread COM graph violations on Windows.
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 width: int = config.FRAME_WIDTH,
                 height: int = config.FRAME_HEIGHT,
                 target_fps: int = config.TARGET_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.actual_fps: float = 0.0
        self.frame_count: int = 0
        self._fps_start_time = time.time()

    def start(self) -> bool:
        """
        Connects to the webcam using DirectShow (with automatic fallback to default backend).
        """
        if self.is_running and self.cap is not None and self.cap.isOpened():
            return True

        # Close any lingering capture
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # 1. Try specified camera index with DirectShow
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        # 2. If specified index failed, try alternative index 1 or 0
        if not self.cap.isOpened():
            alt_index = 1 if self.camera_index == 0 else 0
            self.cap = cv2.VideoCapture(alt_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(alt_index)
            if self.cap.isOpened():
                self.camera_index = alt_index

        if not self.cap.isOpened():
            print(f"[CameraStream Error] Could not open webcam on index {self.camera_index}.")
            return False

        # Set desired resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self.is_running = True
        self.frame_count = 0
        self._fps_start_time = time.time()
        print(f"[CameraStream] Camera opened successfully on index {self.camera_index} ({self.width}x{self.height}).")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next live frame directly from the hardware.
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self._fps_start_time
                if elapsed > 0:
                    self.actual_fps = 30.0 / elapsed
                self._fps_start_time = time.time()
            return True, frame

        return False, None

    def get_fps(self) -> float:
        return self.actual_fps

    def stop(self):
        """Releases the camera hardware cleanly."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[CameraStream] Camera released.")
