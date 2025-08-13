"""
Multi-camera parallel processing for improved performance
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from camera import CameraHandler
from config import Settings, load_camera_config
from detector import YOLODetector

logger = logging.getLogger(__name__)


class MultiCameraProcessor:
    """Handles parallel processing of multiple cameras"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.camera_handler = CameraHandler(settings)
        self.detector = YOLODetector(settings)
        self._lock = threading.Lock()
        self.cameras = load_camera_config(settings)

    def process_camera_worker(self, camera: dict) -> Optional[Dict[str, Any]]:
        """
        Worker function for processing a single camera

        Args:
            camera: Camera configuration dictionary

        Returns:
            Detection results or None if failed
        """
        try:
            # Capture frame
            frame = self.camera_handler.capture_frame_from_rtsp(camera)
            if frame is None:
                return None

            # Run detection
            result = self.detector.detect_objects(frame, camera)
            return result

        except Exception as e:
            logger.error("Error processing camera '%s': %s", camera["name"], e)
            return None

    def process_all_cameras_parallel(self) -> List[Dict[str, Any]]:
        """
        Process all cameras in parallel

        Returns:
            List of detection results
        """
        start_time = time.time()
        results = []

        # Determine optimal number of workers
        max_workers = min(4, len(self.cameras), self.settings.max_concurrent_cameras)

        logger.debug(
            "Processing %d cameras with %d workers",
            len(self.cameras),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all camera processing tasks
            future_to_camera = {
                executor.submit(self.process_camera_worker, camera): camera
                for camera in self.cameras
            }

            # Collect results as they complete
            for future in as_completed(future_to_camera, timeout=60):
                camera = future_to_camera[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per camera
                    if result:
                        results.append(result)
                        logger.debug(
                            "Completed processing for camera '%s'", camera["name"]
                        )
                except Exception as e:
                    logger.error("Error processing camera '%s': %s", camera["name"], e)

        total_time = time.time() - start_time
        logger.info(
            "Completed processing %d cameras in %.2fs (avg %.2fs per camera)",
            len(self.cameras),
            total_time,
            total_time / len(self.cameras) if self.cameras else 0,
        )

        return results

    def process_single_camera(self, camera: dict) -> Optional[Dict[str, Any]]:
        """
        Process a single camera (for sequential processing)

        Args:
            camera: Camera configuration dictionary

        Returns:
            Detection results or None if failed
        """
        return self.process_camera_worker(camera)

    def validate_all_cameras(self) -> Dict[str, bool]:
        """
        Validate all camera connections in parallel

        Returns:
            Dictionary mapping camera IDs to connection status
        """
        results = {}

        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            future_to_camera = {
                executor.submit(
                    self.camera_handler.validate_camera_connection, camera
                ): camera
                for camera in self.cameras
            }

            for future in as_completed(future_to_camera, timeout=30):
                camera = future_to_camera[future]
                try:
                    is_valid = future.result(timeout=10)
                    results[camera["id"]] = is_valid
                    logger.info(
                        "Camera '%s' validation: %s",
                        camera["name"],
                        "OK" if is_valid else "FAILED",
                    )
                except Exception as e:
                    logger.error("Error validating camera '%s': %s", camera["name"], e)
                    results[camera["id"]] = False

        return results

    def cleanup(self):
        """Clean up resources"""
        self.camera_handler.cleanup()
        logger.info("Multi-camera processor cleaned up")


class FrameBuffer:
    """Simple frame buffer for reducing redundant processing"""

    def __init__(self, max_age_seconds: float = 1.0):
        self.max_age = max_age_seconds
        self._buffer = {}
        self._lock = threading.Lock()

    def should_process_frame(self, camera_id: str, frame_hash: int) -> bool:
        """
        Check if frame should be processed based on similarity to recent frames

        Args:
            camera_id: Camera identifier
            frame_hash: Hash of the current frame

        Returns:
            True if frame should be processed, False if it's too similar to recent frames
        """
        current_time = time.time()

        with self._lock:
            # Clean old entries
            self._buffer = {
                cid: (fhash, timestamp)
                for cid, (fhash, timestamp) in self._buffer.items()
                if current_time - timestamp < self.max_age
            }

            # Check if frame is similar to recent one
            if camera_id in self._buffer:
                last_hash, last_time = self._buffer[camera_id]
                if last_hash == frame_hash and current_time - last_time < self.max_age:
                    return False

            # Update buffer
            self._buffer[camera_id] = (frame_hash, current_time)
            return True

    def clear(self):
        """Clear the frame buffer"""
        with self._lock:
            self._buffer.clear()
