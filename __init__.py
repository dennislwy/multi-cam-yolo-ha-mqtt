"""
Multi-Camera YOLO Object Detection with Home Assistant MQTT Integration

A comprehensive system for monitoring multiple RTSP cameras using YOLO object detection
and integrating with Home Assistant via MQTT discovery.
"""

__version__ = "1.0.0"
__author__ = "Dennis Lee"
__email__ = "wylee2000@gmail.com"
__description__ = "Multi-camera YOLO object detection with Home Assistant integration"

from .config import Settings, get_settings
from .monitor import MultiCameraMonitor
from .camera import CameraHandler
from .detector import YOLODetector
from .mqtt_client import MQTTHandler

__all__ = [
    "Settings",
    "get_settings", 
    "MultiCameraMonitor",
    "CameraHandler",
    "YOLODetector",
    "MQTTHandler"
]