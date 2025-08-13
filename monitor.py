"""
Main monitor class that orchestrates camera detection and MQTT publishing
"""

import logging
import time
from typing import List

from camera import CameraHandler
from config import Settings, load_camera_config
from detector import YOLODetector
from mqtt_client import MQTTHandler
from multi_camera_processor import MultiCameraProcessor

logger = logging.getLogger(__name__)


class MultiCameraMonitor:
    """Main monitor class that coordinates all components"""

    def __init__(self, settings: Settings):
        """Initialize the multi-camera monitor with all necessary components.

        Args:
            settings (Settings): Configuration settings for the monitor system.

        Raises:
            ValueError: If no cameras are configured in the settings.
        """
        self.settings = settings

        # Load camera configurations from settings
        self.cameras = load_camera_config(settings)

        # Circuit breaker pattern implementation for camera failure handling
        # Track consecutive failures per camera ID
        self.camera_failure_counts = {}

        # Track when each camera entered circuit breaker state
        self.camera_circuit_breaker_times = {}

        # Threshold before activating circuit breaker
        self.max_consecutive_failures = 3

        # Time to wait before attempting recovery (in minutes)
        self.circuit_breaker_recovery_minutes = 30

        if not self.cameras:
            logger.error("No cameras configured. Exiting.")
            raise ValueError("No cameras configured")

        # Initialize core components for camera handling, detection, and MQTT communication
        self.camera_handler = CameraHandler(settings)
        self.detector = YOLODetector(settings)
        self.mqtt_handler = MQTTHandler(settings)

        # Initialize parallel processor if enabled and multiple cameras are available
        if settings.enable_parallel_processing and len(self.cameras) > 1:
            self.processor = MultiCameraProcessor(settings)
            logger.info("Parallel processing enabled for %d cameras", len(self.cameras))
        else:
            self.processor = None
            logger.info("Sequential processing mode")

        # Validate the complete setup before starting operations
        self._validate_setup()

    def _validate_setup(self):
        """Validate the monitor setup by testing connections and model compatibility.

        This method performs initial validation checks including MQTT connectivity,
        model class support, and logs important system information.
        """
        # Test MQTT broker connection to ensure communication channel is available
        if not self.mqtt_handler.test_connection():
            logger.warning("MQTT connection test failed")

        # Validate that the YOLO model supports all requested detection classes
        unsupported_classes = self.detector.validate_model_classes()
        if unsupported_classes:
            logger.warning(
                "Model doesn't support these classes: %s", unsupported_classes
            )

        # Log detailed model information for debugging and verification
        model_info = self.detector.get_model_info()
        logger.info("Model loaded: %s", model_info.get("model_path", "Unknown"))
        logger.info("Supported classes: %s", model_info.get("supported_classes", []))

    def setup_homeassistant_discovery(self) -> bool:
        """Setup Home Assistant discovery configuration for all cameras.

        Publishes MQTT discovery messages to automatically configure camera entities
        in Home Assistant for seamless integration.

        Returns:
            bool: True if all camera discovery configurations were published successfully,
                 False if any failures occurred.
        """
        logger.info("Setting up Home Assistant discovery")
        successful = self.mqtt_handler.publish_all_discovery_configs(self.cameras)
        return successful == len(self.cameras)

    def run_camera_detection_cycle(self, camera: dict) -> bool:
        """Run a complete detection cycle for a single camera.

        This method handles the full pipeline: frame capture, object detection,
        result publishing, and failure tracking with circuit breaker pattern.

        Args:
            camera (dict): Camera configuration dictionary containing 'id', 'name',
                         'rtsp_url', and other camera-specific settings.

        Returns:
            bool: True if the detection cycle completed successfully, False if any
                 step failed or camera is in circuit breaker state.
        """
        camera_id = camera["id"]

        # Circuit breaker pattern: Skip cameras that have failed too many times
        if self._is_camera_in_circuit_breaker(camera_id):
            if self._should_attempt_recovery(camera_id):
                logger.info(
                    "Attempting recovery for camera '%s' after circuit breaker timeout",
                    camera["name"],
                )
                self._reset_circuit_breaker(camera_id)
            else:
                # Camera is still in circuit breaker cooldown period
                logger.debug(
                    "Camera '%s' is in circuit breaker state, skipping", camera["name"]
                )
                return False

        logger.debug("Starting detection cycle for '%s'", camera["name"])

        try:
            # Step 1: Capture frame from camera's RTSP stream
            frame = self.camera_handler.capture_frame_from_rtsp(camera)
            if frame is None:
                self._record_camera_failure(camera_id, camera["name"])
                return False

            # Step 2: Run YOLO object detection on the captured frame
            detections = self.detector.detect_objects(frame, camera)
            if detections is None:
                self._record_camera_failure(camera_id, camera["name"])
                return False

            # Step 3: Publish detection results to MQTT for Home Assistant
            success = self.mqtt_handler.publish_detection_results(detections, camera)
            if not success:
                logger.warning("Failed to publish results for '%s'", camera["name"])
                return False

            # Reset failure tracking on successful completion of full cycle
            self._reset_camera_failures(camera_id, camera["name"])
            logger.debug("Detection cycle completed for '%s'", camera["name"])
            return True

        except Exception as e:
            logger.error("Unexpected error processing '%s': %s", camera["name"], e)
            self._record_camera_failure(camera_id, camera["name"])
            return False

    def _is_camera_in_circuit_breaker(self, camera_id: str) -> bool:
        """Check if camera is currently in circuit breaker state.

        Args:
            camera_id (str): Unique identifier for the camera.

        Returns:
            bool: True if camera has exceeded failure threshold and is in circuit
                 breaker state, False otherwise.
        """
        return (
            self.camera_failure_counts.get(camera_id, 0)
            >= self.max_consecutive_failures
            and camera_id in self.camera_circuit_breaker_times
        )

    def _should_attempt_recovery(self, camera_id: str) -> bool:
        """Check if enough time has passed to attempt camera recovery.

        Args:
            camera_id (str): Unique identifier for the camera.

        Returns:
            bool: True if the recovery timeout period has elapsed and recovery
                 should be attempted, False if still in cooldown period.
        """
        if camera_id not in self.camera_circuit_breaker_times:
            return False

        # Calculate if enough time has passed since circuit breaker activation
        circuit_breaker_time = self.camera_circuit_breaker_times[camera_id]
        recovery_threshold = circuit_breaker_time + (
            self.circuit_breaker_recovery_minutes * 60
        )  # Convert minutes to seconds
        current_time = time.time()

        return current_time >= recovery_threshold

    def _reset_circuit_breaker(self, camera_id: str):
        """Reset circuit breaker state for a camera to allow operation attempts.

        Args:
            camera_id (str): Unique identifier for the camera to reset.
        """
        # Clear circuit breaker timestamp and reset failure count
        if camera_id in self.camera_circuit_breaker_times:
            del self.camera_circuit_breaker_times[camera_id]
        self.camera_failure_counts[camera_id] = 0

    def _reset_camera_failures(self, camera_id: str, camera_name: str):
        """Reset failure tracking for a camera after successful operation.

        Args:
            camera_id (str): Unique identifier for the camera.
            camera_name (str): Human-readable name of the camera for logging.
        """
        old_failure_count = self.camera_failure_counts.get(camera_id, 0)

        # Reset failure counter to zero
        self.camera_failure_counts[camera_id] = 0

        # Remove from circuit breaker state if camera was there
        if camera_id in self.camera_circuit_breaker_times:
            del self.camera_circuit_breaker_times[camera_id]
            logger.info(
                "Camera '%s' successfully recovered from circuit breaker state",
                camera_name,
            )
        elif old_failure_count > 0:
            logger.debug(
                "Camera '%s' failure count reset after successful operation",
                camera_name,
            )

    def _record_camera_failure(self, camera_id: str, camera_name: str):
        """Record a failure for a camera and implement circuit breaker if threshold reached.

        Args:
            camera_id (str): Unique identifier for the camera.
            camera_name (str): Human-readable name of the camera for logging.
        """
        # Increment failure counter for this camera
        self.camera_failure_counts[camera_id] = (
            self.camera_failure_counts.get(camera_id, 0) + 1
        )

        if self.camera_failure_counts[camera_id] < self.max_consecutive_failures:
            logger.warning(
                "Camera '%s' has failed %d times, threshold is %d",
                camera_name,
                self.camera_failure_counts[camera_id],
                self.max_consecutive_failures,
            )

        # Activate circuit breaker if failure threshold is reached
        if (
            self.camera_failure_counts[camera_id] >= self.max_consecutive_failures
            and camera_id not in self.camera_circuit_breaker_times
        ):

            # Record the timestamp when circuit breaker was activated
            self.camera_circuit_breaker_times[camera_id] = time.time()

            logger.error(
                "Camera '%s' has failed %d times consecutively, enabling circuit breaker for %d minute(s)",
                camera_name,
                self.max_consecutive_failures,
                self.circuit_breaker_recovery_minutes,
            )

    def run_all_cameras_detection_cycle(self) -> bool:
        """Run detection cycle for all configured cameras.

        Processes all cameras either in parallel (if enabled and supported) or
        sequentially, handling failures gracefully and tracking success rates.

        Returns:
            bool: True if at least one camera was processed successfully,
                 False if all cameras failed or no cameras are configured.
        """
        logger.info("Starting detection cycle for all cameras")
        start_time = time.time()

        # Use parallel processing if enabled and processor is available
        if self.processor and self.settings.enable_parallel_processing:
            # Parallel processing path for improved performance
            try:
                results = self.processor.process_all_cameras_parallel()
                successful_cameras = 0

                # Process each parallel result and publish to MQTT
                for result in results:
                    if result:
                        # Find matching camera configuration for this result
                        camera = next(
                            (
                                cam
                                for cam in self.cameras
                                if cam["id"] == result.get("camera_id")
                            ),
                            None,
                        )
                        if camera:
                            # Publish detection results to MQTT broker
                            if self.mqtt_handler.publish_detection_results(
                                result, camera
                            ):
                                successful_cameras += 1
                            else:
                                logger.warning(
                                    "Failed to publish detection for '%s'",
                                    result.get("camera_name", "unknown"),
                                )
                        else:
                            logger.warning(
                                "Camera config not found for result: '%s'",
                                result.get("camera_name", "unknown"),
                            )

                elapsed_time = time.time() - start_time
                logger.info(
                    "Parallel detection cycle completed: %s/%s cameras successful in %.1fs",
                    successful_cameras,
                    len(self.cameras),
                    elapsed_time,
                )

                return successful_cameras > 0

            except Exception as e:
                logger.error(
                    "Parallel processing failed, falling back to sequential: %s", e
                )
                # Fall through to sequential processing as fallback

        # Sequential processing (fallback or when parallel is disabled)
        successful_cameras = 0

        for camera in self.cameras:
            try:
                # Process each camera individually
                if self.run_camera_detection_cycle(camera):
                    successful_cameras += 1

                # Small delay between cameras to prevent system overload
                time.sleep(0.5)  # Reduced delay for better performance

            except Exception as e:
                logger.error("Unexpected error processing '%s': %s", camera["name"], e)

        elapsed_time = time.time() - start_time
        logger.info(
            "Sequential detection cycle completed: %s/%s cameras successful in %.1fs",
            successful_cameras,
            len(self.cameras),
            elapsed_time,
        )

        return successful_cameras > 0

    def run_single_camera_test(self, camera_name: str) -> bool:
        """Run detection test for a specific camera identified by name.

        Args:
            camera_name (str): Name of the camera to test (case-insensitive).

        Returns:
            bool: True if the camera detection test was successful, False if the
                 camera was not found or the test failed.
        """
        # Find camera configuration by name (case-insensitive search)
        camera = next(
            (cam for cam in self.cameras if cam["name"].lower() == camera_name.lower()),
            None,
        )
        if not camera:
            logger.error("Camera '%s' not found", camera_name)
            return False

        logger.info("Running single detection test for '%s'", camera["name"])
        return self.run_camera_detection_cycle(camera)

    def validate_all_cameras(self) -> List[dict]:
        """Validate connections to all configured cameras.

        Tests RTSP connectivity for each camera to ensure they are reachable
        and can provide video streams.

        Returns:
            List[dict]: List of validation results, each containing:
                - camera (str): Camera name
                - valid (bool): Whether connection was successful
                - rtsp_url (str): RTSP URL that was tested
        """
        logger.info("Validating camera connections...")

        # Use parallel validation if available for faster results
        if self.processor and self.settings.enable_parallel_processing:
            try:
                validation_results = self.processor.validate_all_cameras()
                results = []

                # Process parallel validation results
                for camera in self.cameras:
                    is_valid = validation_results.get(camera["id"], False)
                    results.append(
                        {
                            "camera": camera["name"],
                            "valid": is_valid,
                            "rtsp_url": camera["rtsp_url"],
                        }
                    )

                    # Log validation result with visual indicator
                    status = "✓" if is_valid else "✗"
                    logger.info(
                        "%s %s: %s",
                        status,
                        camera["name"],
                        "Connected" if is_valid else "Failed",
                    )

                return results

            except Exception as e:
                logger.error(
                    "Parallel validation failed, falling back to sequential: %s", e
                )

        # Sequential validation (fallback method)
        results = []

        for camera in self.cameras:
            # Test individual camera connection
            is_valid = self.camera_handler.validate_camera_connection(camera)
            results.append(
                {
                    "camera": camera["name"],
                    "valid": is_valid,
                    "rtsp_url": camera["rtsp_url"],
                }
            )

            # Log validation result with visual status indicator
            status = "✓" if is_valid else "✗"
            logger.info(
                "%s %s: %s",
                status,
                camera["name"],
                "Connected" if is_valid else "Failed",
            )

        return results

    def get_system_status(self) -> dict:
        """Get comprehensive system status information.

        Collects status from all system components including cameras, MQTT,
        and detector configuration for monitoring and debugging purposes.

        Returns:
            dict: System status containing:
                - cameras_configured (int): Number of configured cameras
                - camera_names (List[str]): Names of all cameras
                - mqtt_connected (bool): MQTT broker connection status
                - model_info (dict): YOLO model information
                - settings (dict): Key configuration settings
        """
        return {
            "cameras_configured": len(self.cameras),
            "camera_names": [cam["name"] for cam in self.cameras],
            "mqtt_connected": self.mqtt_handler.test_connection(),
            "model_info": self.detector.get_model_info(),
            "settings": {
                "confidence_threshold": self.settings.confidence_threshold,
                "input_size": self.settings.input_size,
                "rtsp_timeout": self.settings.rtsp_timeout,
                "supported_classes": self.settings.supported_classes,
            },
        }

    def cleanup(self):
        """Cleanup all resources and connections before shutdown.

        Properly closes all connections and releases resources to prevent
        memory leaks and ensure clean application termination.
        """
        logger.info("Cleaning up resources...")

        # Cleanup all component resources in proper order
        self.mqtt_handler.cleanup()
        self.camera_handler.cleanup()
        if self.processor:
            self.processor.cleanup()

        logger.info("Cleanup completed")
