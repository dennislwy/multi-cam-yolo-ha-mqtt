"""
YOLO object detection operations
"""

import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from config import Settings

logger = logging.getLogger(__name__)


class YOLODetector:
    """Handles YOLO model operations and object detection"""

    def __init__(self, settings: Settings):
        """Initialize the YOLO detector with configurable device settings.

        Args:
            settings (Settings): Configuration settings including device specification.
        """
        self.settings = settings
        self.model = None
        self._class_ids = None  # Cache for supported class IDs
        self.setup_model()

    def setup_model(self):
        """Initialize YOLO model with optimizations and configurable device."""
        try:
            logger.info("Loading YOLO model: %s", self.settings.yolo_model_path)
            logger.info("Using device: %s", self.settings.device)

            self.model = YOLO(self.settings.yolo_model_path)

            # Warmup the model with a dummy frame to reduce first inference latency
            dummy_frame = np.zeros(
                (self.settings.input_size, self.settings.input_size, 3), dtype=np.uint8
            )
            self.model(
                dummy_frame,
                verbose=False,
                imgsz=self.settings.input_size,
                device=self.settings.device,
            )

            # Cache supported class IDs for faster filtering
            self._get_supported_class_ids()

            logger.info("YOLO model loaded and warmed up successfully")
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            sys.exit(1)

    def detect_objects(
        self, frame: np.ndarray, camera: dict
    ) -> Optional[Dict[str, Any]]:
        """Run YOLO detection on frame using configured device.

        Args:
            frame (np.ndarray): Input image frame for object detection.
            camera (dict): Camera configuration dictionary containing camera metadata.

        Returns:
            Optional[Dict[str, Any]]: Detection results dictionary containing detected objects
                                    and metadata, or None if detection failed.
        """
        # Validate inputs
        if frame is None or frame.size == 0:
            logger.error("Invalid frame provided for detection")
            return None

        if not self.model:
            logger.error("YOLO model not initialized")
            return None

        start_time = time.time()

        try:
            logger.debug("Running detection for '%s'", camera["name"])

            # Run inference with configured device
            results = self.model(
                frame,
                verbose=False,
                imgsz=self.settings.input_size,
                conf=self.settings.confidence_threshold,
                device=self.settings.device,
                half=False,
                max_det=self.settings.max_detection_objects,
                classes=self._get_supported_class_ids(),  # Only detect supported classes
            )

            detection_time = time.time() - start_time

            # Parse results
            detections = {
                "camera_id": camera["id"],
                "camera_name": camera["name"],
                "timestamp": datetime.now().isoformat(),
                "total_objects": 0,
                "detections": [],
            }

            # Initialize class counters
            for class_name in self.settings.supported_classes:
                detections[class_name] = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])

                        # Only count supported classes
                        if class_name in self.settings.supported_classes:
                            detections[class_name] += 1
                            detections["total_objects"] += 1

                            # Store detection details
                            coords = box.xyxy[0].tolist()
                            detections["detections"].append(
                                {
                                    "class": class_name,
                                    "confidence": round(confidence, 2),
                                    "bbox": [round(x, 1) for x in coords],
                                }
                            )

            # Create summary string for logging
            summary = []
            for class_name in self.settings.supported_classes:
                count = detections[class_name]
                if count > 0:
                    summary.append(f"{count} {class_name}{'s' if count > 1 else ''}")

            summary_text = ", ".join(summary) if summary else "no objects"
            logger.info(
                "Detection completed for '%s' in %.2fs on %s: %s",
                camera["name"],
                detection_time,
                self.settings.device,
                summary_text,
            )

            return detections

        except Exception as e:
            detection_time = time.time() - start_time
            logger.error(
                "Error during object detection for '%s' after %.2fs: %s",
                camera["name"],
                detection_time,
                e,
            )
            return None

    def _get_supported_class_ids(self) -> List[int]:
        """
        Get class IDs for supported classes to filter inference

        Returns:
            List of class IDs for supported classes
        """
        if self._class_ids is None:
            if not self.model or not hasattr(self.model, "names"):
                return []

            self._class_ids = [
                class_id
                for class_id, class_name in self.model.names.items()
                if class_name in self.settings.supported_classes
            ]
            logger.info(
                "Cached %d supported class IDs: %s",
                len(self._class_ids),
                self._class_ids,
            )

        return self._class_ids

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded YOLO model including device configuration.

        Returns:
            Dict[str, Any]: Dictionary containing comprehensive model information including
                          device settings, model parameters, and supported classes.
        """
        if not self.model:
            return {}

        return {
            "model_path": self.settings.yolo_model_path,
            "model_type": str(type(self.model)),
            "device": self.settings.device,
            "supported_classes": self.settings.supported_classes,
            "input_size": self.settings.input_size,
            "confidence_threshold": self.settings.confidence_threshold,
            "available_classes": (
                list(self.model.names.values()) if hasattr(self.model, "names") else []
            ),
        }

    def validate_model_classes(self) -> List[str]:
        """
        Validate that supported classes exist in the model

        Returns:
            List of unsupported classes
        """
        if not self.model or not hasattr(self.model, "names"):
            return self.settings.supported_classes

        model_classes = list(self.model.names.values())
        unsupported = [
            cls for cls in self.settings.supported_classes if cls not in model_classes
        ]

        if unsupported:
            logger.warning("Unsupported classes in model: %s", unsupported)
            logger.info("Available model classes: %s", model_classes)

        return unsupported
