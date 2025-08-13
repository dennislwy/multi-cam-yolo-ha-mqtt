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
    """Handles RTSP camera operations with connection caching and retry logic.

    This class manages multiple RTSP camera connections, providing frame capture
    functionality with automatic retry mechanisms and connection validation.

    Attributes:
        VALIDATION_TIMEOUT_MS (int): Timeout in milliseconds for camera validation.
        settings (Settings): Application settings configuration.
        active_captures (dict): Cache of active camera capture objects.
    """

    VALIDATION_TIMEOUT_MS = 5000  # 5 second timeout for validation

    def __init__(self, settings: Settings):
        """Initialize the camera handler.

        Args:
            settings (Settings): Application configuration settings.
        """
        self.settings = settings
        self.active_captures = {}  # Cache camera connections by camera_id

    def capture_frame_from_rtsp(self, camera: dict) -> Optional[np.ndarray]:
        """Capture a single frame from RTSP stream with retry logic.

        Args:
            camera (dict): Camera configuration dictionary containing:
                - id (str): Unique camera identifier
                - name (str): Human-readable camera name
                - rtsp_url (str): RTSP stream URL

        Returns:
            Optional[np.ndarray]: Captured frame as numpy array, or None if capture failed.

        Raises:
            Exception: Any unexpected error during frame capture is caught and logged.
        """
        camera_id = camera["id"]
        start_time = time.time()  # Track capture performance

        try:

            # Get or create capture object from cache
            cap = self._get_or_create_capture(camera_id, camera)
            if not cap:
                return None

            # Attempt frame capture with retry mechanism
            ret, frame = self._read_frame_with_retry(camera_id, camera, cap, 1)

            capture_time = time.time() - start_time  # Calculate total capture time

            if not ret:
                logger.error(
                    "Failed to capture frame from '%s' after retry, took %.2fs",
                    camera["name"],
                    capture_time,
                )
                return None

            logger.info(
                "Frame captured successfully from '%s' in %.2fs",
                camera["name"],
                capture_time,
            )
            return frame

        except Exception as e:
            # Cleanup failed capture to prevent resource leaks
            if camera_id in self.active_captures:
                self.active_captures[camera_id].release()
                del self.active_captures[camera_id]

            capture_time = time.time() - start_time
            logger.error(
                "Error capturing frame from '%s' after %.2fs: %s",
                camera["name"],
                capture_time,
                e,
            )
            return None

    def validate_camera_connection(self, camera: dict) -> bool:
        """Test camera connection without capturing frame for validation purposes.

        Args:
            camera (dict): Camera configuration dictionary containing:
                - name (str): Human-readable camera name
                - rtsp_url (str): RTSP stream URL

        Returns:
            bool: True if camera is accessible and can read frames, False otherwise.

        Raises:
            Exception: Any connection errors are caught and logged as validation failure.
        """
        cap = None
        try:
            # Create temporary capture object for validation
            cap = cv2.VideoCapture(camera["rtsp_url"])

            # Set timeout if available (OpenCV 4.2+)
            if hasattr(cv2, "CAP_PROP_TIMEOUT"):
                cap.set(cv2.CAP_PROP_TIMEOUT, self.VALIDATION_TIMEOUT_MS)

            # Check if stream opened successfully
            if not cap.isOpened():
                return False

            # Test actual frame reading capability
            ret, _ = cap.read()
            return ret

        except Exception as e:
            logger.error("Camera validation failed for '%s': %s", camera["name"], e)
            return False

        finally:
            # Ensure capture is always released to prevent resource leaks
            if cap is not None:
                cap.release()

    def cleanup(self):
        """Release all active camera connections and clear the cache.

        This method should be called when shutting down to properly release
        all OpenCV VideoCapture resources.
        """
        # Release all cached capture objects
        for cap in self.active_captures.values():
            cap.release()
        # Clear the cache dictionary
        self.active_captures.clear()

    def _get_or_create_capture(
        self, camera_id: str, camera: dict
    ) -> Optional[cv2.VideoCapture]:
        """Get existing capture from cache or create new one if needed.

        Args:
            camera_id (str): Unique identifier for the camera.
            camera (dict): Camera configuration dictionary.

        Returns:
            Optional[cv2.VideoCapture]: Active capture object or None if creation failed.
        """
        # Check if capture already exists in cache
        if camera_id not in self.active_captures:
            cap = self._create_capture(camera)
            if not cap:
                return None
            # Cache the new capture object
            self.active_captures[camera_id] = cap

        cap = self.active_captures[camera_id]

        # Validate cached connection is still active
        if not cap.isOpened():
            # Clean up stale connection
            cap.release()
            del self.active_captures[camera_id]
            # Recursively create new connection
            return self._get_or_create_capture(camera_id, camera)

        return cap

    def _create_capture(self, camera: dict) -> Optional[cv2.VideoCapture]:
        """Create and configure new VideoCapture object with optimal settings.

        Args:
            camera (dict): Camera configuration dictionary containing rtsp_url and name.

        Returns:
            Optional[cv2.VideoCapture]: Configured capture object or None if failed.

        Raises:
            Exception: Any errors during capture creation are caught and logged.
        """
        logging.debug("Creating capture for camera '%s'", camera["name"])
        cap = cv2.VideoCapture(camera["rtsp_url"], cv2.CAP_FFMPEG)

        rtsp_timeout_ms = self.settings.rtsp_timeout * 1000

        try:
            # Set multiple timeout properties for better control
            if hasattr(cv2, "CAP_PROP_TIMEOUT"):
                cap.set(cv2.CAP_PROP_TIMEOUT, rtsp_timeout_ms)

            # Set open timeout (FFmpeg specific)
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, rtsp_timeout_ms)

            # Set read timeout (FFmpeg specific)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, rtsp_timeout_ms)

            # Minimize buffer to get most recent frame (reduce latency)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify stream opened successfully
            if not cap.isOpened():
                cap.release()
                logger.error("Failed to open RTSP stream for '%s'", camera["name"])
                return None

            return cap
        except Exception as e:
            # Cleanup on configuration failure
            cap.release()
            logger.error("Failed to create capture for '%s': %s", camera["name"], e)
            return None

    def _read_frame_with_retry(
        self, camera_id: str, camera: dict, cap: cv2.VideoCapture, max_retries: int = 1
    ) -> tuple[bool, Optional[np.ndarray]]:
        """Read frame with configurable retry logic on failure.

        Args:
            camera_id (str): Unique camera identifier for cache management.
            camera (dict): Camera configuration dictionary.
            cap (cv2.VideoCapture): Current capture object.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 1.

        Returns:
            tuple[bool, Optional[np.ndarray]]: Success flag and captured frame array.
                Returns (True, frame) on success, (False, None) on failure.
        """
        logger.debug("Start to capture frame from '%s'", camera["name"])

        # Flush buffer to get most recent frame
        self._flush_buffer(cap, camera["name"], 30)

        # Initial frame read attempt
        ret, frame = cap.read()
        if ret:
            return ret, frame

        # Handle retry attempts for failed reads
        for retry_attempt in range(max_retries):
            logger.warning(
                "Frame capture failed for '%s', retry attempt %d/%d",
                camera["name"],
                retry_attempt + 1,
                max_retries,
            )

            # Clean up failed capture object
            cap.release()
            if camera_id in self.active_captures:
                del self.active_captures[camera_id]

            # Create fresh capture object for retry
            cap = self._create_capture(camera)
            if not cap:
                logger.error(
                    "Failed to open RTSP stream for '%s' on retry %d/%d",
                    camera["name"],
                    retry_attempt + 1,
                    max_retries,
                )
                continue

            # Update cache with new capture object
            try:
                self.active_captures[camera_id] = cap
                ret, frame = cap.read()

                if ret:
                    logger.info(
                        "Frame capture succeeded for '%s' on retry %d/%d",
                        camera["name"],
                        retry_attempt + 1,
                        max_retries,
                    )
                    return ret, frame

            except Exception as e:
                # Clean up on failure
                cap.release()
                del self.active_captures[camera_id]
                logger.error(
                    "Error during retry %d for '%s': %s",
                    retry_attempt + 1,
                    camera["name"],
                    e,
                )

        logger.error(
            "Frame capture failed for '%s' after %d retries",
            camera["name"],
            max_retries,
        )

        return False, None

    def _flush_buffer(
        self, cap: cv2.VideoCapture, camera_name: str, flush_count: int = 5
    ):
        """Flush old frames from buffer to get most recent frame.

        Args:
            cap: VideoCapture object
            camera_name: Camera name for logging
            flush_count: Number of frames to flush (default 5)
        """
        try:
            for i in range(flush_count):
                ret, _ = cap.read()
                if not ret:
                    break
            logger.debug("Flushed %d frames from buffer for '%s'", i + 1, camera_name)
        except Exception as e:
            logger.warning("Error flushing buffer for '%s': %s", camera_name, e)
