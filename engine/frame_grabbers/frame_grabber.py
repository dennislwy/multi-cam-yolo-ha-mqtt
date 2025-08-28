from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class FrameGrabber(ABC):
    """Abstract interface for frame grabber implementations"""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from the buffer

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame) where success indicates
            if a frame was successfully retrieved
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release resources and stop the capture thread"""
        pass

    @abstractmethod
    def isOpened(self) -> bool:
        """Check if the capture is opened and active"""
        pass
