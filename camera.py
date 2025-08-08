"""
Camera operations for RTSP stream handling
"""

import logging
from typing import Optional

import cv2
import numpy as np

from config import Settings

logger = logging.getLogger(__name__)


class CameraHandler:
    """Handles RTSP camera operations"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def capture_frame_from_rtsp(self, camera: dict) -> Optional[np.ndarray]:
        """
        Capture a single frame from RTSP stream

        Args:
            camera: Camera configuration dictionary

        Returns:
            Captured frame as numpy array or None if failed
        """
        try:
            logger.debug("Capturing frame from %s", camera["name"])

            # Create new capture object for each frame to avoid connection issues
            cap = cv2.VideoCapture(camera["rtsp_url"])
            cap.set(
                cv2.CAP_PROP_TIMEOUT, self.settings.rtsp_timeout * 1000
            )  # Convert to milliseconds
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

            if not cap.isOpened():
                logger.error("Failed to open RTSP stream for %s", camera["name"])
                return None

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Failed to capture frame from %s", camera["name"])
                return None

            logger.debug("Frame captured successfully from %s", camera["name"])
            return frame

        except Exception as e:
            logger.error("Error capturing frame from %s: %s", camera["name"], e)
            return None

    def validate_camera_connection(self, camera: dict) -> bool:
        """
        Test camera connection without capturing frame

        Args:
            camera: Camera configuration dictionary

        Returns:
            True if camera is accessible, False otherwise
        """
        try:
            cap = cv2.VideoCapture(camera["rtsp_url"])
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 second timeout for testing

            if not cap.isOpened():
                cap.release()
                return False

            ret, _ = cap.read()
            cap.release()

            return ret

        except Exception as e:
            logger.error("Camera validation failed for %s: %s", camera["name"], e)
            return False
