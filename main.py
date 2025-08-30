import argparse
import logging
import os
import sys
import time
from datetime import datetime
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_settings
from engine.frame_grabbers import (
    LockFreeRingBufferFrameGrabber,
    MultiThreadingFrameGrabber,
    SimpleFrameGrabber,
    SingleFrameGrabber,
)
from engine.object_detection import ObjectDetection

logger: Logger
settings = get_settings()


def setup_logging(settings):
    """Setup logging configuration with daily rotation at midnight.

    Configures logging to rotate daily at midnight and keep only 7 log files.

    Args:
        settings: Application settings containing log configuration.
    """
    # Create log directory if it doesn't exist
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a timed rotating file handler that rotates daily at midnight
    file_handler = TimedRotatingFileHandler(
        filename=settings.log_file,
        when="midnight",  # Rotate at midnight
        interval=1,  # Rotate every 1 day
        backupCount=7,  # Keep 7 backup files (7 days of logs)
        encoding="utf-8",  # Use UTF-8 encoding for log files
    )

    # Set the suffix for rotated files (adds date to filename)
    file_handler.suffix = "%Y-%m-%d"

    # Create console handler for stdout output
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the same format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s [%(name)10.10s][%(funcName)20.20s][%(levelname)5.5s] %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        handlers=[file_handler, console_handler],
    )


def run(
    source,
    model,
    grabber_type: str = "single",
    conf: float = 0.6,
    imgsz: int = 640,
    show: bool = False,
):
    print(f"Using model: {model}")
    print(f"Using source: {source}")
    print(f"Using frame grabber: {grabber_type}")
    print(f"Using confidence threshold: {conf}")
    print(f"Using image size: {imgsz}")

    # Initialize the frame grabber
    if grabber_type == "single":
        grabber = SingleFrameGrabber
    elif grabber_type == "multi":
        grabber = MultiThreadingFrameGrabber
    elif grabber_type == "lockfree":
        grabber = LockFreeRingBufferFrameGrabber
    else:
        raise ValueError(f"Unknown grabber type: {grabber_type}")

    # Load object detection model
    logging.info("Loading model '%s'", model)
    od = ObjectDetection(model)
    logging.info("Model loaded...")
    class_names = od.classes

    cycle_delay = settings.cycle_delay

    # Initialize the frame grabber with the given source
    logging.info("Initializing frame grabber...")
    cap = grabber(source=source)

    try:
        # read a frame every 60s, then perform object detection
        while True:
            grab_time = time.time()
            logging.debug("Grabbing a frame...")
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")

                # Sleep for the remaining cycle time
                delay = max(0, cycle_delay - (time.time() - grab_time))
                logger.debug("Sleeping for the remaining %.2fs", delay)
                time.sleep(delay)

                # Re-initialize the frame grabber
                logging.info("Initializing frame grabber...")
                cap = grabber(source=source)
                continue

            # Display the frame in a window
            if show:
                cv2.imshow(
                    "Object Detection",
                    frame,
                )

            logging.debug("Running detection...")
            start_time = time.time()

            # Perform object detection
            results = od.detect(frame, imgsz=imgsz, conf=conf)

            detection_time = time.time() - start_time

            # Process the results (e.g., display them, send them over a network, etc.)
            process_results(results, class_names, detection_time)

            # Sleep for the remaining cycle time
            delay = max(0, cycle_delay - (time.time() - grab_time))
            logger.debug("Sleeping for the remaining %.2fs", delay)
            time.sleep(delay)

    finally:
        logging.info("Releasing resources...")
        cap.release()
        if show:
            cv2.destroyAllWindows()


def process_results(
    results, class_names: Dict[int, str], detection_time: float
) -> Optional[Dict[str, Any]]:
    # init detections
    detections = {
        "camera_id": "cam1",  # camera["id"],
        "camera_name": "oreo",  # camera["name"],
        "timestamp": datetime.now().isoformat(),
        "total_objects": 0,
        "detections": [],
    }

    class_names_list = list(class_names.values())

    # Initialize class counters and confidence tracking
    class_confidences = {}
    for class_name in class_names_list:
        detections[class_name] = 0
        class_confidences[class_name] = []

    bboxes, class_ids, scores = results
    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        # Get class name
        class_name = class_names[class_id]
        confidence = score
        detections[class_name] += 1
        detections["total_objects"] += 1
        class_confidences[class_name].append(confidence)

        # Store detection details
        detections["detections"].append(
            {
                "class": class_name,
                "confidence": round(confidence, 2),
                "bbox": [round(x, 1) for x in bbox],
            }
        )

    # Create summary string for logging with confidence scores
    summary = []
    for class_name in class_names_list:
        count = detections[class_name]
        if count > 0:
            confidences = class_confidences[class_name]
            conf_str = ", ".join([f"{conf:.2f}" for conf in confidences])
            plural = "s" if count > 1 else ""
            summary.append(f"{count} {class_name}{plural} ({conf_str})")

    summary_text = ", ".join(summary) if summary else "no objects"
    logger.info(
        "Detection completed for in %.2fs: %s",
        detection_time,
        summary_text,
    )

    return detections


def main():
    global logger
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found.")
        return 1

    # Load settings and setup logging
    try:
        load_dotenv()
        setup_logging(settings)
        logger = logging.getLogger(__name__)
        logger.info("Starting Multi-Camera YOLO Detection Monitor")
        logger.info("Python version: %s", sys.version)
        logger.info("Working directory: %s", os.getcwd())

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1

    # args = parse_args()
    # run(args.source, args.model, args.grabber, args.conf, args.imgsz, args.target_fps)

    source = "rtsp://administrator:tapo814@192.168.0.5:554/stream1"  # oreo
    # source = "rtsp://administrator:tapo814@192.168.0.4:554/stream1"  # sofa
    # model = "yolo11n.pt"
    model = settings.yolo_model_path
    conf = 0.6
    imgsz = 640
    show = False

    try:
        run(
            source=source,
            model=model,
            grabber_type="single",
            conf=conf,
            imgsz=imgsz,
            show=show,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nOperation cancelled")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        print("❌ Unexpected error: %s", e)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
