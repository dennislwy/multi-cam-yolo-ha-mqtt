import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_grabber import FrameGrabber


class MultiThreadingFrameGrabber(FrameGrabber):
    def __init__(
        self, source: str, buffer_size: int = 3, target_fps: Optional[float] = None
    ):
        self._source = source
        self._cap = cv2.VideoCapture(self._source)
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        self._frame_count = 0
        self._buffer_lock = threading.Lock()
        self._target_fps = target_fps
        self._last_frame_time = None
        self._ended = False
        self._thread = None

        self._start()

    def _start(self):
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            # Reduce buffer size to minimize latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        """Background thread function to continuously read frames"""
        while not self._ended:
            try:
                # FPS limiting logic
                if self._target_fps and self._target_fps > 0:
                    if self._last_frame_time is not None:
                        elapsed = time.time() - self._last_frame_time
                        min_interval = 1.0 / self._target_fps
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                    self._last_frame_time = time.time()

                ret, frame = self._cap.read()
                if not ret:
                    self._ended = True
                    break
            except cv2.error:
                break

            # If queue is full, remove oldest frame
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()  # Remove oldest frame
                except queue.Empty:
                    pass

            # Add new frame to queue
            try:
                self._frame_queue.put(frame, block=False)
                self._frame_count += 1
            except queue.Full:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self._frame_queue.get(timeout=1.0)
        except queue.Empty:
            # print("No more frame available")
            return False, None

    def release(self) -> None:
        self._ended = True

        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        # Release the video capture
        if self._cap:
            self._cap.release()

        # Clear the queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and not self._ended
