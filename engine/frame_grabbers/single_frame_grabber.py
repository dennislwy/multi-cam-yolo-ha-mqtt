from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_grabber import FrameGrabber


class SingleFrameGrabber(FrameGrabber):
    """
    Frame grabber that opens the video source, reads a single frame, and closes the source each
    time.

    This implementation is simple and ensures that only the latest frame is read from the source,
    but may be slower due to frequent opening and closing of the video stream. It is suitable for
    scenarios where low latency is not critical, CPU usage needs to be minimized and do not
    require high frame rates.
    """

    def __init__(self, source):
        try:
            src = int(source)
        except ValueError:
            src = source
        self._source = src

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            cap = cv2.VideoCapture(self._source)
            ret, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            return ret, frame
        except Exception as e:
            print(f"Error occurred while reading frame: {e}")
            return False, None

    def release(self) -> None:
        pass

    def isOpened(self) -> bool:
        return False
