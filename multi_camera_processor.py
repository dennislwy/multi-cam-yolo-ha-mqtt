"""
Multi-camera parallel processing for improved performance
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from camera import CameraHandler
from config import Settings, load_camera_config
from detector import YOLODetector

if TYPE_CHECKING:
    from monitor import MultiCameraMonitor

logger = logging.getLogger(__name__)


class MultiCameraProcessor:
    """Handles parallel processing of multiple cameras"""

    def __init__(
        self, settings: Settings, monitor: Optional["MultiCameraMonitor"] = None
    ):
        self.settings = settings
        self.camera_handler = CameraHandler(settings)
        self.detector = YOLODetector(settings)
        self._lock = threading.Lock()
        self.cameras = load_camera_config(settings)
        self.monitor = monitor  # Reference to monitor for circuit breaker access

    def process_camera_worker(self, camera: dict) -> Optional[Dict[str, Any]]:
        """
        Worker function for processing a single camera

        Args:
            camera: Camera configuration dictionary

        Returns:
            Detection results or None if failed
        """
        camera_id = camera["id"]

        # Check circuit breaker if monitor is available
        if self.monitor and self.monitor._is_camera_in_circuit_breaker(camera_id):
            if self.monitor._should_attempt_recovery(camera_id):
                logger.debug(
                    "Attempting recovery for camera '%s' after circuit breaker timeout",
                    camera["name"],
                )
                self.monitor._reset_circuit_breaker(camera_id)
            else:
                # Camera is still in circuit breaker cooldown period
                logger.info(
                    "Camera '%s' is in circuit breaker state, skipping (%.1fmin remaining)",
                    camera["name"],
                    self.monitor._recovery_time_remaining(camera_id) / 60,
                )
                return None

        try:
            # Capture frame
            frame = self.camera_handler.capture_frame_from_rtsp(camera)
            if frame is None:
                # Record failure if monitor is available
                if self.monitor:
                    self.monitor._record_camera_failure(camera_id, camera["name"])
                return None

            # Run detection
            result = self.detector.detect_objects(frame, camera)
            if result is None:
                # Record failure if monitor is available
                if self.monitor:
                    self.monitor._record_camera_failure(camera_id, camera["name"])
                return None

            # Reset failures on success if monitor is available
            if self.monitor:
                self.monitor._reset_camera_failures(camera_id, camera["name"])

            return result

        except Exception as e:
            logger.error("Error processing camera '%s': %s", camera["name"], e)
            # Record failure if monitor is available
            if self.monitor:
                self.monitor._record_camera_failure(camera_id, camera["name"])
            return None

    def process_all_cameras_parallel(self) -> List[Dict[str, Any]]:
        """
        Process all cameras in parallel

        Returns:
            List of detection results
        """
        start_time = time.time()
        results = []

        # Filter out cameras in circuit breaker state if monitor is available
        cameras_to_process = []
        if self.monitor:
            for camera in self.cameras:
                if self.monitor._is_camera_in_circuit_breaker(camera["id"]):
                    if self.monitor._should_attempt_recovery(camera["id"]):
                        logger.info(
                            "Including camera '%s' for recovery attempt after circuit breaker timeout",
                            camera["name"],
                        )
                        cameras_to_process.append(camera)
                    else:
                        logger.debug(
                            "Skipping camera '%s' in circuit breaker state (%.1fmin remaining)",
                            camera["name"],
                            self.monitor._recovery_time_remaining(camera["id"]) / 60,
                        )
                else:
                    cameras_to_process.append(camera)
        else:
            cameras_to_process = self.cameras

        if not cameras_to_process:
            logger.warning(
                "No cameras available for processing (all in circuit breaker state)"
            )
            return results

        # Determine optimal number of workers
        max_workers = min(
            4, len(cameras_to_process), self.settings.max_concurrent_cameras
        )

        logger.debug(
            "Processing %d cameras with %d workers (%d skipped due to circuit breaker)",
            len(cameras_to_process),
            max_workers,
            len(self.cameras) - len(cameras_to_process),
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all camera processing tasks
            future_to_camera = {
                executor.submit(self.process_camera_worker, camera): camera
                for camera in cameras_to_process
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
            "Completed processing %d cameras in %.2fs (avg %.2fs per camera, %d total configured)",
            len(cameras_to_process),
            total_time,
            total_time / len(cameras_to_process) if cameras_to_process else 0,
            len(self.cameras),
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
