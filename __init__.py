"""
Multi-Camera YOLO Object Detection with Home Assistant MQTT Integration

A comprehensive system for monitoring multiple RTSP cameras using YOLO object detection
and integrating with Home Assistant via MQTT discovery.
"""

__version__ = "1.0.0"
__author__ = "Dennis Lee"
__email__ = "wylee2000@gmail.com"
__description__ = "Multi-camera YOLO object detection with Home Assistant integration"

import warnings

# Deprecated import - issue warning
try:
    from .camera import CameraHandler

    warnings.warn(
        "CameraHandler is deprecated and will be removed in a future version. "
        "Use RTSPVideoStream instead for better performance and features.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:
    CameraHandler = None

from .config import Settings, get_settings
from .detector import YOLODetector
from .monitor import MultiCameraMonitor
from .mqtt_client import MQTTHandler
from .stream import RTSPVideoStream

__all__ = [
    "Settings",
    "get_settings",
    "MultiCameraMonitor",
    "YOLODetector",
    "MQTTHandler",
    "RTSPVideoStream",
    # Deprecated - will be removed in future version
    "CameraHandler",
]
