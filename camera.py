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

            # Get or create capture object
            cap = self._get_or_create_capture(camera_id, camera)
            if not cap:
                return None

            # Try to read frame with retry logic
            ret, frame = self._read_frame_with_retry(camera_id, camera, cap, 1)

            capture_time = time.time() - start_time

            if not ret:
                logger.error(
                    "Failed to capture frame from %s after retry, took %.2fs",
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
            self._cleanup_failed_capture(camera_id)
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

    def _get_or_create_capture(
        self, camera_id: str, camera: dict
    ) -> Optional[cv2.VideoCapture]:
        """Get existing capture or create new one"""
        if camera_id not in self.active_captures:
            cap = self._create_capture(camera)
            if not cap:
                return None
            self.active_captures[camera_id] = cap

        cap = self.active_captures[camera_id]

        # Check if connection is still valid
        if not cap.isOpened():
            del self.active_captures[camera_id]
            return self._get_or_create_capture(camera_id, camera)

        return cap

    def _create_capture(self, camera: dict) -> Optional[cv2.VideoCapture]:
        """Create and configure new capture object"""
        cap = cv2.VideoCapture(camera["rtsp_url"])

        # Set timeout if available (OpenCV 4.2+)
        if hasattr(cv2, "CAP_PROP_TIMEOUT"):
            cap.set(cv2.CAP_PROP_TIMEOUT, self.settings.rtsp_timeout * 1000)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame

        if not cap.isOpened():
            logger.error("Failed to open RTSP stream for %s", camera["name"])
            return None

        return cap

    def _read_frame_with_retry(
        self, camera_id: str, camera: dict, cap: cv2.VideoCapture, max_retries: int = 1
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Read frame with configurable retry on failure"""
        ret, frame = cap.read()

        if ret:
            return ret, frame

        # Handle retries
        for retry_attempt in range(max_retries):
            logger.warning(
                "Frame capture failed for %s, retry attempt %d/%d",
                camera["name"],
                retry_attempt + 1,
                max_retries,
            )

            # Release failed capture and create new one
            cap.release()
            if camera_id in self.active_captures:
                del self.active_captures[camera_id]

            new_cap = self._create_capture(camera)
            if not new_cap:
                logger.error(
                    "Failed to open RTSP stream for %s on retry %d/%d",
                    camera["name"],
                    retry_attempt + 1,
                    max_retries,
                )
                continue

            # Store new capture and try again
            self.active_captures[camera_id] = new_cap
            ret, frame = new_cap.read()

            if ret:
                logger.info(
                    "Frame capture succeeded for %s on retry %d/%d",
                    camera["name"],
                    retry_attempt + 1,
                    max_retries,
                )
                return ret, frame

            # Update cap reference for next iteration
            cap = new_cap

        logger.error(
            "Frame capture failed for %s after %d retries", camera["name"], max_retries
        )

        return False, None

    def _cleanup_failed_capture(self, camera_id: str):
        """Clean up failed connection"""
        if camera_id in self.active_captures:
            self.active_captures[camera_id].release()
            del self.active_captures[camera_id]
