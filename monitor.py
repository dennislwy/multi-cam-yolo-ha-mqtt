"""
Main monitor class that orchestrates camera detection and MQTT publishing
"""

import logging
import time
from typing import List, Optional

from camera import CameraHandler
from config import Settings, load_camera_config
from detector import YOLODetector
from mqtt_client import MQTTHandler

logger = logging.getLogger(__name__)


class MultiCameraMonitor:
    """Main monitor class that coordinates all components"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cameras = load_camera_config(settings)

        if not self.cameras:
            logger.error("No cameras configured. Exiting.")
            raise ValueError("No cameras configured")

        # Initialize components
        self.camera_handler = CameraHandler(settings)
        self.detector = YOLODetector(settings)
        self.mqtt_handler = MQTTHandler(settings)

        # Validate setup
        self._validate_setup()

    def _validate_setup(self):
        """Validate the monitor setup"""
        # Test MQTT connection
        if not self.mqtt_handler.test_connection():
            logger.warning("MQTT connection test failed")

        # Validate YOLO model classes
        unsupported_classes = self.detector.validate_model_classes()
        if unsupported_classes:
            logger.warning(
                "Model doesn't support these classes: %s", unsupported_classes
            )

        # Log model info
        model_info = self.detector.get_model_info()
        logger.info("Model loaded: %s", model_info.get("model_path", "Unknown"))
        logger.info("Supported classes: %s", model_info.get("supported_classes", []))

    def setup_homeassistant_discovery(self) -> bool:
        """
        Setup Home Assistant discovery for all cameras

        Returns:
            True if all discoveries were published successfully
        """
        logger.info("Setting up Home Assistant discovery")
        successful = self.mqtt_handler.publish_all_discovery_configs(self.cameras)
        return successful == len(self.cameras)

    def run_camera_detection_cycle(self, camera: dict) -> bool:
        """
        Run one complete detection cycle for a specific camera

        Args:
            camera: Camera configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        logger.debug("Starting detection cycle for %s", camera["name"])

        # Capture frame
        frame = self.camera_handler.capture_frame_from_rtsp(camera)
        if frame is None:
            logger.warning(
                "Failed to capture frame from %s, skipping detection", camera["name"]
            )
            return False

        # Run detection
        detections = self.detector.detect_objects(frame, camera)
        if detections is None:
            logger.warning(
                "Detection failed for %s, skipping MQTT publish", camera["name"]
            )
            return False

        # Publish results
        success = self.mqtt_handler.publish_detection_results(detections, camera)
        if not success:
            logger.warning("Failed to publish results for %s", camera["name"])
            return False

        logger.debug("Detection cycle completed for %s", camera["name"])
        return True

    def run_all_cameras_detection_cycle(self) -> bool:
        """
        Run detection cycle for all cameras

        Returns:
            True if at least one camera was processed successfully
        """
        logger.info("Starting detection cycle for all cameras")
        start_time = time.time()

        successful_cameras = 0

        for camera in self.cameras:
            try:
                if self.run_camera_detection_cycle(camera):
                    successful_cameras += 1

                # Small delay between cameras to avoid overwhelming the system
                time.sleep(1)

            except Exception as e:
                logger.error("Unexpected error processing %s: %s", camera["name"], e)

        elapsed_time = time.time() - start_time
        logger.info(
            "Detection cycle completed: %s/%s cameras successful in %.1fs",
            successful_cameras,
            len(self.cameras),
            elapsed_time,
        )

        return successful_cameras > 0

    def run_single_camera_test(self, camera_name: str) -> bool:
        """
        Run detection test for a specific camera by name

        Args:
            camera_name: Name of the camera to test

        Returns:
            True if successful, False otherwise
        """
        camera = next(
            (cam for cam in self.cameras if cam["name"].lower() == camera_name.lower()),
            None,
        )
        if not camera:
            logger.error("Camera '%s' not found", camera_name)
            return False

        logger.info("Running single detection test for %s", camera["name"])
        return self.run_camera_detection_cycle(camera)

    def validate_all_cameras(self) -> List[dict]:
        """
        Validate connections to all cameras

        Returns:
            List of camera validation results
        """
        logger.info("Validating camera connections...")
        results = []

        for camera in self.cameras:
            is_valid = self.camera_handler.validate_camera_connection(camera)
            results.append(
                {
                    "camera": camera["name"],
                    "valid": is_valid,
                    "rtsp_url": camera["rtsp_url"],
                }
            )

            status = "✓" if is_valid else "✗"
            logger.info(
                "%s %s: %s",
                status,
                camera["name"],
                "Connected" if is_valid else "Failed",
            )

        return results

    def get_system_status(self) -> dict:
        """
        Get system status information

        Returns:
            Dictionary containing system status
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
        """Cleanup all resources"""
        logger.info("Cleaning up resources...")
        self.mqtt_handler.cleanup()
        logger.info("Cleanup completed")
