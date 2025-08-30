import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_grabber import FrameGrabber

logger = logging.getLogger(__name__)


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
        cap: Optional[cv2.VideoCapture] = None

        try:
            cap = cv2.VideoCapture(self._source)
            ret, frame = cap.read() if cap.isOpened() else (False, None)
            return ret, frame
        except Exception as e:
            logging.error("Error occurred while reading frame: %s", e)
            return False, None
        finally:
            if cap is not None:
                cap.release()

    def release(self) -> None:
        pass

    def isOpened(self) -> bool:
        return False
