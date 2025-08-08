"""
Configuration settings for multi-camera YOLO detection system
"""

import os
import sys
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Main configuration settings"""

    # MQTT Configuration
    mqtt_broker: str = "192.168.0.88"
    mqtt_port: int = 1883
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    discovery_prefix: str = "homeassistant"

    # YOLO Model Configuration
    yolo_model_path: str = "yolov8n.pt"
    input_size: int = 320
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)
    supported_classes: List[str] = ["person", "dog", "poop"]

    # System Configuration
    device_name: str = "camera_monitor"
    log_file: str = "/var/log/camera_monitor.log"
    log_level: str = "INFO"
    rtsp_timeout: int = Field(default=10, ge=1, le=60)

    # Camera Configuration
    camera_count: int = Field(ge=1)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("supported_classes")
    @classmethod
    def validate_classes(cls, v: List[str]) -> List[str]:
        """Validate supported classes"""
        if not v or len(v) == 0:
            raise ValueError("supported_classes cannot be empty")
        return v

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


def load_camera_config(settings: Settings) -> List[dict]:
    """Load camera configuration from environment variables"""
    cameras = []

    for i in range(1, settings.camera_count + 1):
        name = os.getenv(f"CAMERA_{i}_NAME")
        rtsp_url = os.getenv(f"CAMERA_{i}_RTSP_URL")

        if not name or not rtsp_url:
            print(f"Camera {i} missing required configuration (NAME or RTSP_URL)")
            continue

        camera = {
            "id": i,
            "name": name,
            "rtsp_url": rtsp_url,
            "location": os.getenv(f"CAMERA_{i}_LOCATION", ""),
            "enabled": os.getenv(f"CAMERA_{i}_ENABLED", "true").lower() == "true",
        }

        if camera["enabled"]:
            cameras.append(camera)
            print(f"Loaded camera: {name} ({rtsp_url})")

    return cameras


def get_settings() -> Settings:
    """Get validated settings instance"""
    try:
        return Settings()
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


def create_example_env_file():
    """Create an example .env file"""
    example_content = """# Multi-Camera YOLO Detection Configuration

# MQTT Configuration
MQTT_BROKER=192.168.0.88
MQTT_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
DISCOVERY_PREFIX=homeassistant

# YOLO Model Configuration
YOLO_MODEL_PATH=yolov8n.pt
INPUT_SIZE=320
CONFIDENCE_THRESHOLD=0.6
SUPPORTED_CLASSES=person,dog,poop

# System Configuration
DEVICE_NAME=camera_monitor
LOG_FILE=/var/log/camera_monitor.log
LOG_LEVEL=INFO
RTSP_TIMEOUT=10

# Camera Configuration
CAMERA_COUNT=3

# Camera 1
CAMERA_1_NAME=Front Door
CAMERA_1_RTSP_URL=rtsp://administrator:password@192.168.0.5:554/stream1
CAMERA_1_LOCATION=Front Entrance
CAMERA_1_ENABLED=true

# Camera 2
CAMERA_2_NAME=Backyard
CAMERA_2_RTSP_URL=rtsp://administrator:password@192.168.0.6:554/stream1
CAMERA_2_LOCATION=Backyard
CAMERA_2_ENABLED=true

# Camera 3
CAMERA_3_NAME=Living Room
CAMERA_3_RTSP_URL=rtsp://administrator:password@192.168.0.7:554/stream1
CAMERA_3_LOCATION=Living Room
CAMERA_3_ENABLED=false
"""

    with open(".env.example", "w", encoding="utf-8") as f:
        f.write(example_content)

    print("Created .env.example file. Copy it to .env and modify with your settings.")
