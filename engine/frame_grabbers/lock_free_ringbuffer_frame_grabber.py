import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_grabber import FrameGrabber


class LockFreeRingBuffer:
    def __init__(self, size: int):
        self._size = size
        self._buffer = [None] * size
        self._write_idx = 0
        self._read_idx = 0
        self._not_empty = threading.Event()

    def put(self, item):
        next_write = (self._write_idx + 1) % self._size
        if next_write == self._read_idx:
            # Buffer full, overwrite oldest
            self._read_idx = (self._read_idx + 1) % self._size
        self._buffer[self._write_idx] = item
        self._write_idx = next_write
        self._not_empty.set()

    def get(self, timeout: float = 1.0):
        start = time.time()
        while self._read_idx == self._write_idx:
            waited = time.time() - start
            if waited >= timeout:
                return None
            self._not_empty.wait(timeout - waited)
            self._not_empty.clear()
        item = self._buffer[self._read_idx]
        self._read_idx = (self._read_idx + 1) % self._size
        return item

    def empty(self):
        return self._read_idx == self._write_idx

    def clear(self):
        self._write_idx = 0
        self._read_idx = 0
        self._buffer = [None] * self._size
        self._not_empty.clear()


class LockFreeRingBufferFrameGrabber(FrameGrabber):
    def __init__(
        self,
        source,
        buffer_size: int = 8,
        target_fps: Optional[float] = None,
    ):
        try:
            src = int(source)
        except ValueError:
            src = source
        self._source = src
        self._cap = cv2.VideoCapture(self._source)
        self._buffer = LockFreeRingBuffer(buffer_size)
        self._target_fps = target_fps
        self._last_frame_time = None
        self._ended = False
        self._thread = None
        self._start()

    def _start(self):
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        while not self._ended:
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
            self._buffer.put(frame)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame = self._buffer.get()
        if frame is not None:
            return True, frame
        return False, None

    def release(self) -> None:
        self._ended = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._buffer.clear()

    def isOpened(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and not self._ended
