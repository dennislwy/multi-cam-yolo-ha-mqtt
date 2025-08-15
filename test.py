import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import get_context

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# ========================
# Setup logging
# ========================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)10.10s][%(funcName)20.20s][%(levelname)5.5s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ========================
# Load config
# ========================
load_dotenv()

PERSISTENT_RTSP = os.getenv("PERSISTENT_RTSP", "False").lower() == "true"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11m.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu").lower()
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "60"))
EXPORT_RESULTS = os.getenv("EXPORT_RESULTS", "False").lower() == "true"
RTSP_STREAMS = [
    url.strip() for url in os.getenv("RTSP_STREAMS", "").split(",") if url.strip()
]

if not RTSP_STREAMS:
    raise ValueError("No RTSP streams defined in .env file!")

# Create results folder if exporting
if EXPORT_RESULTS:
    os.makedirs("results", exist_ok=True)
    # Wipe all files in results folder on startup
    for filename in os.listdir("results"):
        file_path = os.path.join("results", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    logger.info("Cleared results folder on startup")

# ========================
# Frame grabbing
# ========================
persistent_caps = []


def init_persistent_connections():
    global persistent_caps
    logger.info("Initializing persistent RTSP connections...")
    persistent_caps = []
    for url in RTSP_STREAMS:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not cap.isOpened():
            logger.warning("Cannot open stream: %s", url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        persistent_caps.append(cap)


def get_frame(rtsp_url, cam_idx=None):
    start_time = time.time()

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    if not cap.isOpened():
        if cam_idx is not None:
            logger.warning("[CAM %s] Cannot open stream: %s", cam_idx + 1, rtsp_url)
        else:
            logger.warning("Cannot open stream: %s", rtsp_url)
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    cap.release()

    elapsed = time.time() - start_time
    if not ret:
        if cam_idx is not None:
            logger.warning("[CAM %s] Failed to grab frame: %s", cam_idx + 1, rtsp_url)
        else:
            logger.warning("Failed to grab frame: %s", rtsp_url)
        return None
    if cam_idx is not None:
        logger.debug("[CAM %s] Frame grab took %.2fs", cam_idx + 1, elapsed)
    else:
        logger.debug("Frame grab took %.2fs for %s", elapsed, rtsp_url)
    return frame


def get_frame_persistent(index):
    start_time = time.time()
    if index >= len(persistent_caps) or persistent_caps[index] is None:
        logger.warning("Invalid persistent camera index: %s", index)
        return None

    cap = persistent_caps[index]

    # Check if connection is still valid
    if not cap.isOpened():
        logger.warning(
            "[CAM %s] Connection lost, attempting to reconnect...", index + 1
        )
        # Try to reconnect
        rtsp_url = RTSP_STREAMS[index]
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        persistent_caps[index] = cap

        if not cap.isOpened():
            logger.error("[CAM %s] Failed to reconnect to stream", index + 1)
            return None

    ret, frame = cap.read()
    elapsed = time.time() - start_time

    if not ret:
        logger.warning(
            "[CAM %s] Failed to grab frame from persistent camera - attempting reconnection",
            index + 1,
        )
        # Try immediate reconnection
        rtsp_url = RTSP_STREAMS[index]
        cap.release()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        persistent_caps[index] = cap

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info(
                    "[CAM %s] Successfully reconnected and grabbed frame", index + 1
                )
            else:
                logger.error("[CAM %s] Reconnected but still no frame", index + 1)
        return frame if ret else None

    logger.debug(
        "[CAM %s] Frame grab took %.2fs (skipped %d frames)",
        index + 1,
        elapsed,
        frames_skipped,
    )
    return frame


# ========================
# YOLO detection
# ========================
yolo_model = None


def init_yolo():
    """Load YOLO model for each process (CPU mode)."""
    global yolo_model
    if yolo_model is None:
        logger.info("Loading YOLO model in worker...")
        start_time = time.time()
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to(YOLO_DEVICE)
        elapsed = time.time() - start_time
        logger.debug("YOLO model loaded in worker in %.2fs", elapsed)


def run_yolo(frame, cam_idx=None):
    if frame is None:
        return None
    start_time = time.time()
    global yolo_model
    results = yolo_model(frame)
    elapsed = time.time() - start_time
    if cam_idx is not None:
        logger.debug("[CAM %s] YOLO detection took %.2fs", cam_idx + 1, elapsed)
    else:
        logger.debug("YOLO detection took %.2fs", elapsed)
    return results[0]


def run_yolo_with_index(args):
    """Helper function for multiprocessing with camera index."""
    cam_idx, frame = args
    return run_yolo(frame, cam_idx)


# ========================
# Export results
# ========================
def save_detection_image(result, frame, cam_idx):
    """Save frame with bounding boxes if detections > 0."""
    if result is None or len(result.boxes) == 0:
        return
    # Draw detections on frame
    annotated_frame = result.plot()
    # Save to results folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/cam{cam_idx+1}_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_frame)
    logger.info("Saved detection image: %s", filename)


# ========================
# Main
# ========================
if __name__ == "__main__":
    if PERSISTENT_RTSP:
        init_persistent_connections()

    if YOLO_DEVICE == "gpu":
        # GPU mode → load YOLO once in main process, no multiprocessing
        logger.info("Using GPU mode — single process YOLO")
        start_time = time.time()
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to("cuda")
        elapsed = time.time() - start_time
        logger.info("YOLO model loaded on GPU in %.2fs", elapsed)
        while True:
            start_time = time.time()

            # Grab frames
            if PERSISTENT_RTSP:
                frames = [get_frame_persistent(i) for i in range(len(RTSP_STREAMS))]
            else:
                with ThreadPoolExecutor(max_workers=len(RTSP_STREAMS)) as tpe:
                    frames = list(
                        tpe.map(
                            lambda args: get_frame(args[1], args[0]),
                            enumerate(RTSP_STREAMS),
                        )
                    )

            # Run YOLO sequentially on GPU
            detection_start = time.time()
            detections = [
                run_yolo(f, i) if f is not None else None for i, f in enumerate(frames)
            ]
            detection_elapsed = time.time() - detection_start
            logger.debug("Total GPU YOLO detection took %.2fs", detection_elapsed)

            # Display & save results
            for cam_idx, result in enumerate(detections):
                if result is None:
                    logger.info("[CAM %s] No frame/detection", cam_idx + 1)
                else:
                    logger.info(
                        "[CAM %s] Detections: %s objects",
                        cam_idx + 1,
                        len(result.boxes),
                    )
                    if EXPORT_RESULTS:
                        save_detection_image(result, frames[cam_idx], cam_idx)

            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL_SECONDS - elapsed)
            logger.info(
                "Cycle complete in %.2fs, sleeping %.2fs...", elapsed, sleep_time
            )
            time.sleep(sleep_time)

    else:
        # CPU mode → multiprocessing for parallel inference
        logger.info("Using CPU mode — multiprocessing enabled")
        ctx = get_context("spawn")
        with ctx.Pool(processes=len(RTSP_STREAMS), initializer=init_yolo) as pool:
            while True:
                start_time = time.time()

                # Grab frames
                if PERSISTENT_RTSP:
                    frames = [get_frame_persistent(i) for i in range(len(RTSP_STREAMS))]
                else:
                    with ThreadPoolExecutor(max_workers=len(RTSP_STREAMS)) as tpe:
                        frames = list(
                            tpe.map(
                                lambda args: get_frame(args[1], args[0]),
                                enumerate(RTSP_STREAMS),
                            )
                        )

                # Run YOLO in parallel
                detection_start = time.time()
                detections = list(pool.map(run_yolo_with_index, enumerate(frames)))
                detection_elapsed = time.time() - detection_start
                logger.debug("Total CPU YOLO detection took %.2fs", detection_elapsed)

                # Display & save results
                for cam_idx, result in enumerate(detections):
                    if result is None:
                        logger.info("[CAM %s] No frame/detection", cam_idx + 1)
                    else:
                        logger.info(
                            "[CAM %s] Detections: %s objects",
                            cam_idx + 1,
                            len(result.boxes),
                        )
                        if EXPORT_RESULTS:
                            save_detection_image(result, frames[cam_idx], cam_idx)

                elapsed = time.time() - start_time
                sleep_time = max(0, INTERVAL_SECONDS - elapsed)
                logger.info(
                    "Cycle complete in %.2fs, sleeping %.2fs...", elapsed, sleep_time
                )
                time.sleep(sleep_time)
