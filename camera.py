"""
Camera operations for RTSP stream handling
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from config import Settings

logger = logging.getLogger(__name__)


class CameraHandler:
    """Handles RTSP camera operations"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.active_captures = {}  # Cache camera connections

    def capture_frame_from_rtsp(self, camera: dict) -> Optional[np.ndarray]:
        """
        Capture a single frame from RTSP stream

        Args:
            camera: Camera configuration dictionary

        Returns:
            Captured frame as numpy array or None if failed
        """
        try:
            start_time = time.time()
            camera_id = camera["id"]

            # Reuse existing capture or create new one
            if camera_id not in self.active_captures:
                # Create new capture object
                cap = cv2.VideoCapture(camera["rtsp_url"])

                # Set timeout if available (OpenCV 4.2+)
                if hasattr(cv2, "CAP_PROP_TIMEOUT"):
                    cap.set(cv2.CAP_PROP_TIMEOUT, self.settings.rtsp_timeout * 1000)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

                if not cap.isOpened():
                    logger.error("Failed to open RTSP stream for %s", camera["name"])
                    return None

                self.active_captures[camera_id] = cap

            cap = self.active_captures[camera_id]

            # Check if connection is still valid
            if not cap.isOpened():
                # Reconnect
                del self.active_captures[camera_id]
                return self.capture_frame_from_rtsp(camera)

            # Read frame
            ret, frame = cap.read()

            capture_time = time.time() - start_time

            if not ret:
                logger.error(
                    "Failed to capture frame from %s after %.2fs",
                    camera["name"],
                    capture_time,
                )
                return None

            logger.info(
                "Frame captured successfully from %s in %.2fs",
                camera["name"],
                capture_time,
            )
            return frame

        except Exception as e:
            # Clean up failed connection
            if camera_id in self.active_captures:
                self.active_captures[camera_id].release()
                del self.active_captures[camera_id]

            # Log error with capture time
            capture_time = time.time() - start_time
            logger.error(
                "Error capturing frame from %s after %.2fs: %s",
                camera["name"],
                capture_time,
                e,
            )
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

            # Set timeout if available
            if hasattr(cv2, "CAP_PROP_TIMEOUT"):
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

    def cleanup(self):
        """Release all camera connections"""
        for cap in self.active_captures.values():
            cap.release()
        self.active_captures.clear()
