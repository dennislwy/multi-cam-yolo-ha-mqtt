"""
Main monitor class that orchestrates camera detection and MQTT publishing
"""

import logging
import time
from typing import Dict, List

from config import Settings, load_camera_config
from detector import YOLODetector
from mqtt_client import MQTTHandler
from multi_camera_processor import MultiCameraProcessor
from stream import RTSPVideoStream

logger = logging.getLogger(__name__)


class MultiCameraMonitor:
    """Main monitor class that coordinates all components"""

    def __init__(self, settings: Settings, output_results: bool = False):
        """Initialize the multi-camera monitor with all necessary components.

        Args:
            settings (Settings): Configuration settings for the monitor system.
            output_results (bool): Whether to save detected result images to output folder.

        Raises:
            ValueError: If no cameras are configured in the settings.
        """
        self.settings = settings
        self.output_results = output_results

        # Load camera configurations from settings
        self.cameras = load_camera_config(settings)

        # Circuit breaker pattern implementation for camera failure handling
        # Track consecutive failures per camera ID
        self.camera_failure_counts = {}

        # Track when each camera entered circuit breaker state
        self.camera_circuit_breaker_times = {}

        # Threshold before activating circuit breaker
        self.max_consecutive_failures = (
            settings.circuit_breaker_max_consecutive_failures
        )

        # Time to wait before attempting recovery (in minutes)
        self.circuit_breaker_recovery_minutes = (
            settings.circuit_breaker_recovery_minutes
        )

        if not self.cameras:
            logger.error("No cameras configured. Exiting.")
            raise ValueError("No cameras configured")

        # Initialize core components for detection and MQTT communication
        self.detector = YOLODetector(settings, output_results=self.output_results)
        self.mqtt_handler = MQTTHandler(settings)

        # Initialize video streams for each camera
        self.camera_streams: Dict[str, RTSPVideoStream] = {}
        self._initialize_camera_streams()

        # Initialize parallel processor if enabled and multiple cameras are available
        if settings.enable_parallel_processing and len(self.cameras) > 1:
            self.processor = MultiCameraProcessor(
                settings, monitor=self, output_results=self.output_results
            )
            logger.info("Parallel processing enabled for %d cameras", len(self.cameras))
        else:
            self.processor = None
            logger.info("Sequential processing mode")

        # Validate the complete setup before starting operations
        self._validate_setup()

    def _initialize_camera_streams(self):
        """Initialize RTSPVideoStream objects for all configured cameras."""
        logger.info("Initializing camera streams...")

        for camera in self.cameras:
            camera_id = camera["id"]
            rtsp_url = camera["rtsp_url"]

            try:
                # Create RTSPVideoStream with settings-based configuration
                stream = RTSPVideoStream(
                    rtsp_url=rtsp_url,
                    reconnect_delay=self.settings.rtsp_timeout,
                    max_reconnect_attempts=5,
                    target_fps=self.settings.rtsp_target_fps,
                )

                # Start the stream
                stream.start()
                self.camera_streams[camera_id] = stream

                logger.info("Initialized stream for camera '%s'", camera["name"])

            except Exception as e:
                logger.error(
                    "Failed to initialize stream for camera '%s': %s", camera["name"], e
                )
                # Continue with other cameras even if one fails

    def cleanup_camera_streams(self):
        """Clean up all camera streams when shutting down."""
        logger.info("Cleaning up camera streams...")

        for camera_id, stream in self.camera_streams.items():
            try:
                stream.stop()
                logger.debug("Stopped stream for camera ID: %s", camera_id)
            except Exception as e:
                logger.error("Error stopping stream for camera ID %s: %s", camera_id, e)

        self.camera_streams.clear()

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
        logger.info("YOLO model loaded: %s", model_info.get("model_path", "Unknown"))
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
                logger.info(
                    "Camera '%s' is in circuit breaker state, skipping (%.1fmin remaining)",
                    camera["name"],
                    self._recovery_time_remaining(camera_id) / 60,
                )
                return False

        start_time = time.time()
        logger.debug("Starting detection cycle for '%s'", camera["name"])

        try:
            # Step 1: Capture frame from camera's RTSP stream using RTSPVideoStream
            stream = self.camera_streams.get(camera_id)
            if stream is None or not stream.is_running():
                logger.error(
                    "Camera stream for '%s' is not available or running", camera["name"]
                )
                self._record_camera_failure(camera_id, camera["name"])
                return False

            frame = stream.read()
            if frame is None:
                logger.warning("No frame available from camera '%s'", camera["name"])
                self._record_camera_failure(camera_id, camera["name"])
                return False

            # Step 2: Run YOLO object detection on the captured frame
            detections = self.detector.detect_objects(frame, camera)
            if detections is None:
                self._record_camera_failure(camera_id, camera["name"])
                return False

            # Camera and detection successful - reset failure tracking
            self._reset_camera_failures(camera_id, camera["name"])

            # Step 3: Publish detection results to MQTT for Home Assistant
            # MQTT failures should NOT trigger camera circuit breaker
            success = self.mqtt_handler.publish_detection_results(detections, camera)
            if not success:
                logger.warning(
                    "Failed to publish MQTT results for '%s' (camera still operational)",
                    camera["name"],
                )
                # Return True because camera/detection worked, only MQTT failed
                detection_time = time.time() - start_time
                logger.info(
                    "Detection cycle completed for '%s' in %.2f seconds (but MQTT publish failed)",
                    camera["name"],
                    detection_time,
                )
                return True

            detection_time = time.time() - start_time
            logger.info(
                "Detection cycle completed for '%s' in %.2f seconds",
                camera["name"],
                detection_time,
            )
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

    def _recovery_time_remaining(self, camera_id: str) -> float:
        """Calculate remaining recovery time for a camera in circuit breaker state.

        Args:
            camera_id (str): Unique identifier for the camera.

        Returns:
            float: Remaining recovery time in seconds, or 0 if not in circuit breaker state.
        """
        if camera_id not in self.camera_circuit_breaker_times:
            return 0.0

        circuit_breaker_time = self.camera_circuit_breaker_times[camera_id]
        recovery_threshold = circuit_breaker_time + (
            self.circuit_breaker_recovery_minutes * 60
        )
        current_time = time.time()

        return max(0.0, recovery_threshold - current_time)

    def _should_attempt_recovery(self, camera_id: str) -> bool:
        """Check if enough time has passed to attempt camera recovery.

        Args:
            camera_id (str): Unique identifier for the camera.

        Returns:
            bool: True if the recovery timeout period has elapsed and recovery
                 should be attempted, False if still in cooldown period.
        """
        remaining_time_sec = self._recovery_time_remaining(camera_id)
        return remaining_time_sec <= 0

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
                "Camera '%s' has failed %d times consecutively, enabling circuit breaker for %dmin",
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
        logger.debug("Starting detection cycle for all cameras")
        start_time = time.time()

        # Use parallel processing if enabled and processor is available
        if self.processor and self.settings.enable_parallel_processing:
            # Parallel processing path for improved performance
            try:
                results = self.processor.process_all_cameras_parallel()
                successful_cameras = 0
                mqtt_failures = 0

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
                            # Camera/detection was successful, count it regardless of MQTT
                            successful_cameras += 1

                            # Publish detection results to MQTT broker
                            # MQTT failures don't affect camera success count
                            if not self.mqtt_handler.publish_detection_results(
                                result, camera
                            ):
                                mqtt_failures += 1
                                logger.warning(
                                    "Failed to publish MQTT results for '%s' (camera detection successful)",
                                    result.get("camera_name", "unknown"),
                                )
                        else:
                            logger.warning(
                                "Camera config not found for result: '%s'",
                                result.get("camera_name", "unknown"),
                            )
                    # Note: Failed cameras (None results) are already handled in the worker

                elapsed_time = time.time() - start_time
                if mqtt_failures > 0:
                    logger.info(
                        "Parallel detection cycle completed: %s/%s cameras successful, %s MQTT failures in %.1fs",
                        successful_cameras,
                        len(self.cameras),
                        mqtt_failures,
                        elapsed_time,
                    )
                else:
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
        self.cleanup_camera_streams()  # Updated to use new camera streams
        if self.processor:
            self.processor.cleanup()

        logger.info("Cleanup completed")
