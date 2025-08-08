"""
MQTT client operations for Home Assistant integration
"""

import json
import logging
import sys
from typing import Any, Dict, List

import paho.mqtt.client as mqtt

from config import Settings

logger = logging.getLogger(__name__)


class MQTTHandler:
    """Handles MQTT operations for Home Assistant integration"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.setup_client()

    def setup_client(self):
        """Setup MQTT client and connection"""
        try:
            self.client = mqtt.Client()

            # Set credentials if provided
            if self.settings.mqtt_username and self.settings.mqtt_password:
                self.client.username_pw_set(
                    self.settings.mqtt_username, self.settings.mqtt_password
                )

            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish

            # Connect to broker
            logger.info(
                f"Connecting to MQTT broker {self.settings.mqtt_broker}:{self.settings.mqtt_port}"
            )
            self.client.connect(self.settings.mqtt_broker, self.settings.mqtt_port, 60)
            self.client.loop_start()

        except Exception as e:
            logger.error(f"Failed to setup MQTT: {e}")
            sys.exit(1)

    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")

    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning("Disconnected from MQTT broker")

    def on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logger.debug(f"Message {mid} published successfully")

    def get_camera_topics(self, camera: dict) -> Dict[str, str]:
        """
        Get MQTT topics for a specific camera

        Args:
            camera: Camera configuration dictionary

        Returns:
            Dictionary containing topic information
        """
        camera_id = f"camera_{camera['id']}"
        unique_id = f"{self.settings.device_name}_{camera_id}_obj_detection"
        sensor_topic = f"{self.settings.discovery_prefix}/sensor/{unique_id}"

        return {
            "unique_id": unique_id,
            "config_topic": f"{sensor_topic}/config",
            "state_topic": f"{sensor_topic}/state",
        }

    def publish_discovery_config(self, camera: dict) -> bool:
        """
        Publish Home Assistant discovery configuration for a specific camera

        Args:
            camera: Camera configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            topics = self.get_camera_topics(camera)

            config = {
                "name": f"{camera['name']} Object Detection",
                "unique_id": topics["unique_id"],
                "state_topic": topics["state_topic"],
                "value_template": "{{ value_json.total_objects }}",
                "json_attributes_topic": topics["state_topic"],
                "json_attributes_template": "{{ value_json | tojson }}",
                "device": {
                    "identifiers": [f"{self.settings.device_name}_{camera['id']}"],
                    "name": f"{camera['name']} Monitor",
                    "model": "YOLO Object Detection",
                    "manufacturer": "Custom",
                },
                "icon": "mdi:camera-account",
            }

            # Add location if specified
            if camera.get("location"):
                config["device"]["suggested_area"] = camera["location"]

            result = self.client.publish(
                topics["config_topic"], json.dumps(config), retain=True
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published discovery config for camera: {camera['name']}")
                return True
            else:
                logger.error(
                    f"Failed to publish discovery config for {camera['name']}: {result.rc}"
                )
                return False

        except Exception as e:
            logger.error(f"Error publishing discovery config for {camera['name']}: {e}")
            return False

    def publish_all_discovery_configs(self, cameras: List[dict]) -> int:
        """
        Publish Home Assistant discovery configuration for all cameras

        Args:
            cameras: List of camera configuration dictionaries

        Returns:
            Number of successful publications
        """
        successful = 0
        for camera in cameras:
            if self.publish_discovery_config(camera):
                successful += 1

        logger.info(
            f"Published discovery configs: {successful}/{len(cameras)} successful"
        )
        return successful

    def publish_detection_results(
        self, detections: Dict[str, Any], camera: dict
    ) -> bool:
        """
        Publish detection results for a specific camera to MQTT

        Args:
            detections: Detection results dictionary
            camera: Camera configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            topics = self.get_camera_topics(camera)
            payload = json.dumps(detections)

            result = self.client.publish(topics["state_topic"], payload, qos=1)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Detection results published for {camera['name']}")
                return True
            else:
                logger.error(
                    f"Failed to publish to MQTT for {camera['name']}: {result.rc}"
                )
                return False

        except Exception as e:
            logger.error(f"Error publishing to MQTT for {camera['name']}: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test MQTT connection

        Returns:
            True if connected, False otherwise
        """
        return self.client.is_connected() if self.client else False

    def cleanup(self):
        """Cleanup MQTT resources"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        logger.debug("MQTT cleanup completed")
