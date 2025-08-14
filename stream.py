"""
RTSPVideoStream - Multi-threaded RTSP Stream Handler with Recording Capabilities

This module provides a robust, thread-safe solution for handling RTSP video streams
with automatic reconnection, snapshot capture, and video recording functionality.

Author: Dennis Lee
License: MIT

Dependencies:
    - opencv-python (cv2)
    - numpy
    - threading (built-in)
    - logging (built-in)

Example Usage:
    Basic stream reading:
        >>> stream = RTSPVideoStream("rtsp://user:pass@camera.ip/stream")
        >>> stream.start()
        >>> frame = stream.read()
        >>> stream.stop()

    Context manager (recommended):
        >>> with RTSPVideoStream("rtsp://user:pass@camera.ip/stream") as stream:
        ...     frame = stream.read()
        ...     if frame is not None:
        ...         cv2.imshow("Stream", frame)

    Recording video:
        >>> with RTSPVideoStream("rtsp://camera.ip/stream") as stream:
        ...     # Record for 30 seconds
        ...     path = stream.start_recording(max_duration=30)
        ...     while stream.is_recording():
        ...         time.sleep(1)

    Taking snapshots:
        >>> with RTSPVideoStream("rtsp://camera.ip/stream") as stream:
        ...     snapshot_path = stream.save_snapshot()
        ...     print(f"Saved to: {snapshot_path}")
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RTSPVideoStream:
    """A class to read frames from an RTSP stream in a separate thread.

    This prevents the main thread from blocking and ensures we always get the latest frame.
    The class handles automatic reconnection on stream failures and provides thread-safe
    frame access.

    Attributes:
        rtsp_url (str): The RTSP URL to connect to.
        reconnect_delay (int): Delay in seconds between reconnection attempts.
        max_reconnect_attempts (int): Maximum number of reconnection attempts (-1 for unlimited).
        _reconnect_count (int): Current number of reconnection attempts.
        _cap (cv2.VideoCapture): OpenCV VideoCapture object.
        _frame (numpy.ndarray): Latest frame from the stream.
        _stopped (bool): Flag to control the reading thread.
        _thread (threading.Thread): Background thread for frame reading.
        _lock (threading.Lock): Thread lock for frame access synchronization.
        _recording (bool): Flag to indicate if recording is active.
        _video_writer (cv2.VideoWriter): OpenCV VideoWriter for recording.
        _recording_lock (threading.Lock): Thread lock for recording synchronization.
        _recording_start_time (datetime): Timestamp when recording started.
        _current_recording_path (str): Path to the current recording file.
    """

    TIMEOUT_CONNECT = 10000
    TIMEOUT_READ = 5000
    DEFAULT_FPS = 20.0

    def __init__(
        self, rtsp_url: str, reconnect_delay: int = 5, max_reconnect_attempts: int = -1
    ) -> None:
        """Initialize the VideoStream object.

        Args:
            rtsp_url (str): The RTSP URL to connect to.
            reconnect_delay (int, optional): Delay in seconds between reconnection
                attempts. Defaults to 5.
            max_reconnect_attempts (int, optional): Maximum number of reconnection
                attempts. Use -1 for unlimited attempts. Defaults to -1.
        """
        if not rtsp_url or not isinstance(rtsp_url, str):
            raise ValueError("rtsp_url must be a non-empty string")
        if reconnect_delay < 0:
            raise ValueError("reconnect_delay must be non-negative")

        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_count = 0
        self._cap = None
        self._frame = None
        self._stopped = False
        self._thread = None

        # Thread lock to ensure frame access is thread-safe
        self._lock = threading.Lock()

        # Recording attributes
        self._recording = False
        self._video_writer = None
        self._recording_lock = threading.Lock()
        self._recording_start_time = None
        self._current_recording_path = None

        # Attempt initial connection to validate the stream
        connect_success = self._connect()

        if not connect_success:
            logger.warning(
                "Initial connection to '%s' failed. Will retry when started.", rtsp_url
            )

    def start(self):
        """Start the background thread for frame reading.

        Creates and starts a daemon thread that continuously reads frames from
        the RTSP stream. If a thread is already running, this method does nothing.

        Raises:
            RuntimeError: If thread creation fails.
        """
        if self._thread is not None and self._thread.is_alive():
            # Prevent multiple threads from being started simultaneously
            logger.warning("Stream is already running")
            return

        self._stopped = False
        # Create daemon thread so it doesn't prevent program exit
        self._thread = threading.Thread(target=self._update, args=())
        self._thread.daemon = True
        self._thread.start()

    def read(self) -> Optional[np.ndarray]:
        """Get the most recent frame from the stream.

        Returns:
            numpy.ndarray or None: A copy of the latest frame, or None if no frame
                is available. Returns a copy to prevent external modifications.
        """
        # Thread-safe frame access with lock
        with self._lock:
            if self._frame is not None and self._frame.size > 0:
                return self._frame.copy()
            return None

    def stop(self):
        """Stop the background thread and release all resources.

        This method gracefully shuts down the stream by:
        - Setting the stop flag for the background thread
        - Stopping any active recording
        - Waiting for the thread to finish (with timeout)
        - Releasing the OpenCV VideoCapture object
        """
        # Stop any active recording first
        if self._recording:
            self.stop_recording()

        self._stopped = True
        # Wait for background thread to finish with timeout to prevent hanging
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)  # 10 second timeout

        # Clean up video capture resources
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_running(self) -> bool:
        """Check if the stream is running and healthy.

        Returns:
            bool: True if the stream is active and functioning properly, False otherwise.
                A stream is considered running if the thread is alive, not stopped,
                and the video capture is open.
        """
        return (
            not self._stopped
            and self._thread is not None
            and self._thread.is_alive()
            and self._cap is not None
            and self._cap.isOpened()
        )

    # region Snapshot

    def save_snapshot(
        self, filename: Optional[str] = None, directory: str = "snapshots"
    ) -> Optional[str]:
        """Save the current frame as a snapshot image.

        Args:
            filename (str, optional): Custom filename for the snapshot. If None,
                generates a timestamp-based filename. Should include file extension.
            directory (str, optional): Directory to save the snapshot. Defaults to "snapshots".
                Directory will be created if it doesn't exist.

        Returns:
            str or None: The full path of the saved snapshot file, or None if saving failed.

        Raises:
            ValueError: If no frame is available to save.
        """
        # Get current frame thread-safely
        current_frame = self.read()

        if current_frame is None:
            raise ValueError("No frame available to save as snapshot")

        # Create directory if it doesn't exist
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create snapshot directory '%s': %s", directory, e)
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :-3
            ]  # Include milliseconds
            # Extract camera identifier from RTSP URL for filename
            url_parts = self.rtsp_url.split("/")
            camera_id = url_parts[-1] if url_parts[-1] else "camera"
            # Remove any invalid filename characters
            camera_id = "".join(c for c in camera_id if c.isalnum() or c in ("-", "_"))
            filename = f"{camera_id}_{timestamp}.jpg"

        # Ensure filename has extension
        if not any(
            filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]
        ):
            filename += ".jpg"

        # Full path for the snapshot
        filepath = os.path.join(directory, filename)

        try:
            # Save the frame as an image
            success = cv2.imwrite(filepath, current_frame)
            if success:
                logger.info("Snapshot saved: %s", filepath)
                return filepath
            else:
                logger.error("Failed to save snapshot: %s", filepath)
                return None
        except Exception as e:
            logger.error("Exception while saving snapshot: %s", e)
            return None

    # endregion

    # region Recording

    def start_recording(
        self,
        filename: Optional[str] = None,
        directory: str = "recordings",
        codec: str = "mp4v",
        fps: float = DEFAULT_FPS,
        max_duration: Optional[int] = None,
    ) -> Optional[str]:
        """Start recording the video stream to a file.

        Args:
            filename (str, optional): Custom filename for the recording. If None,
                generates a timestamp-based filename.
            directory (str, optional): Directory to save recordings. Defaults to "recordings".
            codec (str, optional): Video codec to use. Defaults to "mp4v".
            fps (float, optional): Frames per second for recording. Defaults to 20.0.
            max_duration (int, optional): Maximum recording duration in seconds.
                If None, records until manually stopped.

        Returns:
            str or None: The full path of the recording file, or None if start failed.

        Raises:
            RuntimeError: If recording is already in progress or no frame is available.
        """
        if not 1.0 <= fps <= 60.0:
            raise ValueError("FPS must be between 1.0 and 60.0")

        with self._recording_lock:
            if self._recording:
                raise RuntimeError("Recording is already in progress")

            # Check if we have a frame to determine video properties
            current_frame = self.read()
            if current_frame is None:
                raise RuntimeError("No frame available to start recording")

            # Create directory if it doesn't exist
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                logger.error(
                    "Failed to create recording directory '%s': %s", directory, e
                )
                return None

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                url_parts = self.rtsp_url.split("/")
                camera_id = url_parts[-1] if url_parts[-1] else "camera"
                camera_id = "".join(
                    c for c in camera_id if c.isalnum() or c in ("-", "_")
                )
                filename = f"{camera_id}_{timestamp}.avi"

            # Ensure filename has proper extension
            if not any(
                filename.lower().endswith(ext) for ext in [".avi", ".mp4", ".mov"]
            ):
                filename += ".avi"

            # Full path for the recording
            self._current_recording_path = os.path.join(directory, filename)

            # Get frame dimensions
            height, width = current_frame.shape[:2]

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self._video_writer = cv2.VideoWriter(
                self._current_recording_path, fourcc, fps, (width, height)
            )

            if not self._video_writer.isOpened():
                logger.error("Failed to open video writer for recording")
                self._video_writer = None
                return None

            self._recording = True
            self._recording_start_time = datetime.now()

            # Schedule automatic stop if max_duration is specified
            if max_duration is not None:

                def auto_stop():
                    time.sleep(max_duration)
                    if self._recording:
                        self.stop_recording()
                        logger.info(
                            "Recording automatically stopped after %d seconds",
                            max_duration,
                        )

                auto_stop_thread = threading.Thread(target=auto_stop)
                auto_stop_thread.daemon = True
                auto_stop_thread.start()

            logger.info("Recording started: %s", self._current_recording_path)
            return self._current_recording_path

    def stop_recording(self) -> Optional[Dict[str, Union[str, float]]]:
        """Stop the current recording.

        Returns:
            dict or None: Recording information containing:
                - 'filepath': Path to the recorded file
                - 'duration': Recording duration in seconds
                - 'start_time': Recording start timestamp
                - 'end_time': Recording end timestamp
            Returns None if no recording was in progress.
        """
        with self._recording_lock:
            if not self._recording:
                logger.warning("No recording in progress to stop")
                return None

            self._recording = False

            # Calculate recording duration
            end_time = datetime.now()
            duration = (end_time - self._recording_start_time).total_seconds()

            # Release the video writer
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None

            recording_info = {
                "filepath": self._current_recording_path,
                "duration": duration,
                "start_time": self._recording_start_time.isoformat(),
                "end_time": end_time.isoformat(),
            }

            logger.info(
                "Recording stopped: %s (Duration: %.2f seconds)",
                self._current_recording_path,
                duration,
            )

            self._current_recording_path = None
            self._recording_start_time = None

            return recording_info

    def is_recording(self) -> bool:
        """Check if recording is currently active.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording

    def get_recording_info(self) -> Optional[Dict[str, Union[str, float]]]:
        """Get information about the current recording.

        Returns:
            dict or None: Current recording information containing:
                - 'filepath': Path to the recording file
                - 'duration': Current recording duration in seconds
                - 'start_time': Recording start timestamp
            Returns None if no recording is in progress.
        """
        if not self._recording or self._recording_start_time is None:
            return None

        current_duration = (datetime.now() - self._recording_start_time).total_seconds()

        return {
            "filepath": self._current_recording_path,
            "duration": current_duration,
            "start_time": self._recording_start_time.isoformat(),
        }

    # endregion

    def get_status(self) -> Dict[str, Union[bool, int]]:
        """Get detailed status information about the stream.

        Returns:
            dict: A dictionary containing detailed status information with keys:
                - 'running': Whether the stream is not stopped
                - 'thread_alive': Whether the background thread is alive
                - 'cap_opened': Whether the video capture is open
                - 'reconnect_count': Number of reconnection attempts
                - 'has_frame': Whether a frame is currently available
                - 'recording': Whether recording is active
                - 'recording_duration': Current recording duration (if recording)
        """
        status = {
            "running": not self._stopped,
            "thread_alive": self._thread is not None and self._thread.is_alive(),
            "cap_opened": self._cap is not None and self._cap.isOpened(),
            "reconnect_count": self._reconnect_count,
            "has_frame": self._frame is not None,
            "recording": self._recording,
            "healthy": self.is_running(),
        }

        # Add recording duration if currently recording
        if self._recording and self._recording_start_time:
            duration = (datetime.now() - self._recording_start_time).total_seconds()
            status["recording_duration"] = duration

        return status

    def _connect(self) -> bool:
        """Establish connection to the RTSP stream.

        This private method handles the low-level connection establishment including:
        - Creating the OpenCV VideoCapture object
        - Setting timeout parameters
        - Reading an initial frame to validate the connection
        - Configuring buffer settings for low-latency streaming

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            logger.info("Attempting to connect to '%s'", self.rtsp_url)

            # Use FFMPEG backend for better RTSP support
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Configure timeout settings to prevent hanging
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.TIMEOUT_CONNECT)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.TIMEOUT_READ)

            # Verify that the stream opened successfully
            if not self._cap.isOpened():
                logger.error("Could not connect to stream '%s'", self.rtsp_url)
                return False

            # Set buffer size to 1 to minimize latency and get most recent frames
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read initial frame to validate the stream is providing data
            ret, frame = self._cap.read()
            if not ret:
                logger.error("Could not read initial frame from stream")
                return False

            # Store the initial frame thread-safely
            with self._lock:
                self._frame = frame
            logger.info("Successfully connected to stream '%s'", self.rtsp_url)
            return True

        except Exception as e:
            logger.error("Exception during connection: {%s}", e)
            # Clean up on connection failure
            if self._cap:
                self._cap.release()
                self._cap = None
            return False

    def _update(self):
        """Main loop for the background thread that reads frames continuously.

        This method runs in a separate thread and handles:
        - Continuous frame reading from the RTSP stream
        - Automatic reconnection on stream failures
        - Reconnection attempt limiting
        - Thread-safe frame storage
        - Recording frames when recording is active
        """
        while not self._stopped:
            # Check if connection is lost or not established
            if not self._cap or not self._cap.isOpened():
                # Enforce reconnection attempt limits if configured
                if (
                    self.max_reconnect_attempts > 0
                    and self._reconnect_count >= self.max_reconnect_attempts
                ):
                    logger.warning(
                        "Max reconnection attempts (%d) reached. Stopping.",
                        self.max_reconnect_attempts,
                    )
                    self._stopped = True
                    break

                logger.warning(
                    "Stream lost. Attempting to reconnect... (attempt %d)",
                    self._reconnect_count + 1,
                )

                # Clean up existing connection before reconnecting
                if self._cap:
                    self._cap.release()

                # Attempt reconnection
                if self._connect():
                    self._reconnect_count = 0  # Reset counter on successful connection
                    logger.info(
                        "Successfully reconnected to stream '%s'", self.rtsp_url
                    )
                else:
                    # Increment counter and wait before next attempt
                    self._reconnect_count += 1
                    time.sleep(self.reconnect_delay)
                continue

            # Attempt to read a frame from the stream
            ret, frame = self._cap.read()

            if ret:
                # Add frame validation before storing
                if frame is not None and frame.size > 0:
                    # Successfully read frame - store it thread-safely
                    with self._lock:
                        self._frame = frame

                    # Write frame to recording if active
                    with self._recording_lock:
                        if self._recording and self._video_writer is not None:
                            try:
                                self._video_writer.write(frame)
                            except Exception as e:
                                logger.error("Error writing frame to recording: %s", e)

                    self._reconnect_count = 0  # Reset on successful read
                else:
                    logger.warning("Received invalid frame from stream")
            else:
                # Frame read failed - trigger reconnection sequence
                logger.warning(
                    "Failed to read frame from stream '%s'. Attempting to reconnect...",
                    self.rtsp_url,
                )

                # Release the connection to trigger reconnection logic
                if self._cap:
                    self._cap.release()
                    self._cap = None

    def __enter__(self) -> "RTSPVideoStream":
        """Enter the context manager - start the stream.

        Returns:
            VideoStream: The VideoStream instance for use in the context.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit the context manager - clean up resources.

        Args:
            exc_type: Exception type (if any).
            exc_val: Exception value (if any).
            exc_tb: Exception traceback (if any).

        Returns:
            bool: False to indicate exceptions should not be suppressed.
        """
        self.stop()
        return False


if __name__ == "__main__":
    import sys

    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)10.10s][%(funcName)15.15s][%(levelname)5.5s] %(message)s",
        handlers=[
            # Console handler for real-time monitoring
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific log levels for different components
    logger.setLevel(logging.INFO)

    # Ask user to enter RTSP URL instead of hardcoding credentials
    print("RTSP Video Stream Test with Recording")
    print("=" * 40)

    # Get RTSP URL from user input
    url = input(
        "Enter RTSP URL (e.g., rtsp://username:password@ip:port/stream): "
    ).strip()

    # Validate input
    if not url:
        print("Error: RTSP URL cannot be empty")
        sys.exit(1)

    if not url.lower().startswith("rtsp://"):
        print("Warning: URL should start with 'rtsp://'")
        confirm = input("Continue anyway? (y/N): ").lower()
        if confirm != "y":
            sys.exit(1)

    try:
        # Use context manager for automatic cleanup
        with RTSPVideoStream(url) as stream:
            print("Stream started with context manager!")

            # Wait for stream to initialize
            time.sleep(2)

            if not stream.is_running():
                raise ConnectionError("Failed to start stream.")

            # Start a test recording
            recording_duration = 10
            logger.info("Starting %d seconds test recording...", recording_duration)
            recording_path = stream.start_recording(max_duration=recording_duration)
            if recording_path:
                logger.info("Recording to: %s", recording_path)

            while stream.is_recording():
                try:
                    # Non-blocking input simulation - in real app you'd use proper input handling
                    time.sleep(1)

                    rec_info = stream.get_recording_info()
                    if rec_info:
                        logger.info("Recording info: %s", rec_info)

                except ValueError:
                    logger.error("No frame available, waiting...")
                    time.sleep(1)

            snapshot_count = 0
            max_snapshots = 100
            logger.info("Taking %d snapshots every 1 second...", max_snapshots)

            while snapshot_count < max_snapshots:
                try:
                    if stream.is_running():
                        # Take snapshot with custom directory structure
                        file_path = stream.save_snapshot()

                        if file_path:
                            snapshot_count += 1
                            logger.info("Snapshot #%d: %s", snapshot_count, file_path)

                        # Check stream status periodically
                        if snapshot_count % 5 == 0:  # Every 5 snapshots
                            status = stream.get_status()
                            logger.info("Stream status: %s", status)

                    time.sleep(1)

                except ValueError:
                    logger.error("No frame available, waiting...")
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
