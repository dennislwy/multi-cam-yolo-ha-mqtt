"""
Multi-Camera YOLO Object Detection with Home Assistant MQTT Integration

Main entry point for the application
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import create_example_env_file, get_settings
from monitor import MultiCameraMonitor


def setup_logging(settings):
    """Setup logging configuration"""
    # Create log directory if it doesn't exist
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(name)10.10s][%(funcName)20.20s][%(levelname)5.5s] %(message)s",
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Camera YOLO Detection Monitor for Home Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run single detection cycle (for cron)
  %(prog)s --test                    # Test all cameras once
  %(prog)s --test --camera "Front Door"  # Test specific camera
  %(prog)s --continuous              # Run continuous monitoring
  %(prog)s --validate                # Validate camera connections
  %(prog)s --status                  # Show system status
  %(prog)s --create-env              # Create example .env file
        """,
    )

    parser.add_argument("--test", action="store_true", help="Run single detection test")
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuous monitoring (for testing)",
    )
    parser.add_argument(
        "--create-env", action="store_true", help="Create example .env file"
    )
    parser.add_argument(
        "--camera", type=str, help="Test specific camera by name (use with --test)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate camera connections"
    )
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument(
        "--setup-ha", action="store_true", help="Setup Home Assistant discovery only"
    )

    args = parser.parse_args()

    # Handle special commands that don't require full setup
    if args.create_env:
        create_example_env_file()
        return 0

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found.")
        print("Run with --create-env to create an example file.")
        return 1

    # Load settings and setup logging
    try:
        settings = get_settings()
        load_dotenv()
        setup_logging(settings)
        logger = logging.getLogger(__name__)
        logger.info("Starting Multi-Camera YOLO Detection Monitor")
        logger.info("Python version: %s", sys.version)
        logger.info("Working directory: %s", os.getcwd())

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1

    # Initialize monitor
    try:
        monitor = MultiCameraMonitor(settings)
        logger.info("Monitor initialized with %s cameras", len(monitor.cameras))

    except Exception as e:
        logger.error("Failed to initialize monitor: %s", e)
        return 1

    try:
        # Handle different command modes
        if args.status:
            # Show system status
            status = monitor.get_system_status()
            print("\n📊 System Status:")
            print(f"  Cameras configured: {status['cameras_configured']}")
            print(f"  Camera names: {', '.join(status['camera_names'])}")
            print(f"  MQTT connected: {'✓' if status['mqtt_connected'] else '✗'}")
            print(f"  Model: {status['model_info'].get('model_path', 'Unknown')}")
            print(
                f"  Supported classes: {', '.join(status['settings']['supported_classes'])}"
            )
            print(
                f"  Confidence threshold: {status['settings']['confidence_threshold']}"
            )
            return 0

        elif args.validate:
            # Validate camera connections
            results = monitor.validate_all_cameras()
            print("\n📹 Camera Validation Results:")
            for result in results:
                status_icon = "✓" if result["valid"] else "✗"
                print(f"  {status_icon} {result['camera']}")

            failed_cameras = [r for r in results if not r["valid"]]
            if failed_cameras:
                print(f"\n⚠️  {len(failed_cameras)} camera(s) failed connection test")
                return 1
            else:
                print(f"\n✅ All {len(results)} cameras validated successfully")
                return 0

        elif args.setup_ha:
            # Setup Home Assistant discovery only
            success = monitor.setup_homeassistant_discovery()
            if success:
                print("✅ Home Assistant discovery setup completed")
                return 0
            else:
                print("❌ Home Assistant discovery setup failed")
                return 1

        elif args.test:
            if args.camera:
                # Test specific camera
                success = monitor.run_single_camera_test(args.camera)
            else:
                # Test all cameras
                logger.info("Running single detection test for all cameras")
                success = monitor.run_all_cameras_detection_cycle()

            if success:
                print("✅ Test completed successfully")
                return 0
            else:
                print("❌ Test failed")
                return 1

        elif args.continuous:
            # Continuous monitoring mode
            logger.info("Starting continuous monitoring mode")
            print("🔄 Running continuous monitoring... (Press Ctrl+C to stop)")

            try:
                while True:
                    monitor.run_all_cameras_detection_cycle()
                    logging.debug(
                        "Waiting for next cycle...sleep for %d seconds",
                        settings.cycle_delay,
                    )
                    print(
                        "⏳ Waiting for next cycle...sleep for %d seconds"
                        % settings.cycle_delay
                    )
                    time.sleep(settings.cycle_delay)  # Use configurable delay
            except KeyboardInterrupt:
                logger.info("Continuous monitoring stopped by user")
                print("\n⏹️  Monitoring stopped")
                return 0

        else:
            # Default: single run (for cron job)
            logger.info("Running single detection cycle")
            success = monitor.run_all_cameras_detection_cycle()
            return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nOperation cancelled")
        return 0

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        print("❌ Unexpected error: %s", e)
        return 1

    finally:
        # Cleanup resources
        try:
            monitor.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
