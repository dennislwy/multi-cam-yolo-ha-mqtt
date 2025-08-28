import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_grabber import FrameGrabber


class SimpleFrameGrabber(FrameGrabber):
    """
    Frame grabber that opens the video source once and reads frames sequentially.

    This implementation keeps the video stream open, allowing for sequential frame access.
    It is suitable for simple use cases where multi-threading is not required, and the
    overhead of opening and closing the video stream for each frame is undesirable.

    Optionally, a target frame rate can be specified to limit how often frames are read.
    """

    def __init__(self, source, target_fps: Optional[float] = None):
        self._source = source
        self._cap = cv2.VideoCapture(self._source)
        self._last_frame_time = None
        self._target_fps = target_fps

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # FPS limiting logic
        if self._target_fps and self._target_fps > 0:
            if self._last_frame_time is not None:
                elapsed = time.time() - self._last_frame_time
                min_interval = 1.0 / self._target_fps
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            self._last_frame_time = time.time()

        if self.isOpened():
            return self._cap.read()
        return False, None

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def isOpened(self) -> bool:
        return self._cap.isOpened()
