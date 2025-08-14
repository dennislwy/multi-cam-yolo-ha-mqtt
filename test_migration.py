#!/usr/bin/env python3
"""
Test script to verify CameraHandler to RTSPVideoStream migration

This script tests the basic functionality of the migrated monitor system
to ensure that RTSPVideoStream integration works correctly.
"""

import logging
import sys
import time
import warnings
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_deprecated_import():
    """Test that CameraHandler import shows deprecation warning."""
    logger.info("Testing deprecated CameraHandler import...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            from camera import CameraHandler

            # Check if deprecation warning was issued
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]

            if deprecation_warnings:
                logger.info("✓ Deprecation warning correctly shown for CameraHandler")
                return True
            else:
                logger.warning("✗ No deprecation warning shown for CameraHandler")
                return False

        except ImportError as e:
            logger.error("✗ Failed to import CameraHandler: %s", e)
            return False


def test_rtsp_video_stream_import():
    """Test that RTSPVideoStream can be imported successfully."""
    logger.info("Testing RTSPVideoStream import...")

    try:
        from stream import RTSPVideoStream

        logger.info("✓ RTSPVideoStream imported successfully")
        return True
    except ImportError as e:
        logger.error("✗ Failed to import RTSPVideoStream: %s", e)
        return False


def test_monitor_initialization():
    """Test that MultiCameraMonitor can be initialized without CameraHandler."""
    logger.info("Testing MultiCameraMonitor initialization...")

    try:
        from config import get_settings
        from monitor import MultiCameraMonitor

        # Create minimal settings for testing
        settings = get_settings()

        # Mock camera configuration to avoid actual camera connections
        test_cameras = [
            {
                "id": "test_camera_1",
                "name": "Test Camera 1",
                "rtsp_url": "rtsp://administrator:tapo814@192.168.0.4:554/stream1",
                "enabled": True,
            }
        ]

        # Temporarily override camera config
        import config

        original_load_camera_config = config.load_camera_config
        config.load_camera_config = lambda _: test_cameras

        try:
            # This should not raise any import errors
            monitor = MultiCameraMonitor(settings, output_results=False)
            logger.info("✓ MultiCameraMonitor initialized successfully")

            # Check that camera_streams attribute exists
            if hasattr(monitor, "camera_streams"):
                logger.info("✓ camera_streams attribute found")
            else:
                logger.warning("✗ camera_streams attribute missing")
                return False

            # Check that cleanup method exists
            if hasattr(monitor, "cleanup_camera_streams"):
                logger.info("✓ cleanup_camera_streams method found")
            else:
                logger.warning("✗ cleanup_camera_streams method missing")
                return False

            # Test cleanup
            monitor.cleanup()
            logger.info("✓ Monitor cleanup completed successfully")

            return True

        except Exception as e:
            logger.error("✗ Failed to initialize MultiCameraMonitor: %s", e)
            return False
        finally:
            # Restore original function
            config.load_camera_config = original_load_camera_config

    except ImportError as e:
        logger.error("✗ Failed to import required modules: %s", e)
        return False


def test_multi_camera_processor():
    """Test that MultiCameraProcessor works without CameraHandler."""
    logger.info("Testing MultiCameraProcessor...")

    try:
        from config import get_settings
        from multi_camera_processor import MultiCameraProcessor

        settings = get_settings()

        # Mock camera configuration
        test_cameras = [
            {
                "id": "test_camera_1",
                "name": "Test Camera 1",
                "rtsp_url": "rtsp://administrator:tapo814@192.168.0.4:554/stream1",
            }
        ]

        import config

        original_load_camera_config = config.load_camera_config
        config.load_camera_config = lambda _: test_cameras

        try:
            processor = MultiCameraProcessor(settings)
            logger.info("✓ MultiCameraProcessor initialized successfully")

            # Check that camera_handler attribute is removed
            if not hasattr(processor, "camera_handler"):
                logger.info("✓ camera_handler dependency removed")
            else:
                logger.warning("✗ camera_handler still present")
                return False

            # Test cleanup
            processor.cleanup()
            logger.info("✓ Processor cleanup completed successfully")

            return True

        except Exception as e:
            logger.error("✗ Failed to initialize MultiCameraProcessor: %s", e)
            return False
        finally:
            config.load_camera_config = original_load_camera_config

    except ImportError as e:
        logger.error("✗ Failed to import MultiCameraProcessor: %s", e)
        return False


def run_migration_tests() -> bool:
    """Run all migration tests and return overall success status."""
    logger.info("Starting CameraHandler to RTSPVideoStream migration tests...")
    logger.info("=" * 60)

    tests = [
        ("Deprecated Import Test", test_deprecated_import),
        ("RTSPVideoStream Import Test", test_rtsp_video_stream_import),
        ("Monitor Initialization Test", test_monitor_initialization),
        ("MultiCameraProcessor Test", test_multi_camera_processor),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 40)

        try:
            result = test_func()
            results.append(result)

            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")

        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append(False)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(results)
    total = len(results)

    logger.info(f"Tests passed: {passed}/{total}")

    if all(results):
        logger.info("🎉 ALL TESTS PASSED - Migration successful!")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED - Migration needs attention")
        return False


if __name__ == "__main__":
    success = run_migration_tests()
    sys.exit(0 if success else 1)
