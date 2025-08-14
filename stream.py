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
    """

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
        - Waiting for the thread to finish (with timeout)
        - Releasing the OpenCV VideoCapture object
        """
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

    def get_status(self) -> Dict[str, Union[bool, int]]:
        """Get detailed status information about the stream.

        Returns:
            dict: A dictionary containing detailed status information with keys:
                - 'running': Whether the stream is not stopped
                - 'thread_alive': Whether the background thread is alive
                - 'cap_opened': Whether the video capture is open
                - 'reconnect_count': Number of reconnection attempts
                - 'has_frame': Whether a frame is currently available
        """
        return {
            "running": not self._stopped,
            "thread_alive": self._thread is not None and self._thread.is_alive(),
            "cap_opened": self._cap is not None and self._cap.isOpened(),
            "reconnect_count": self._reconnect_count,
            "has_frame": self._frame is not None,
        }

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
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

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

    # Ask user to enter RTSP URL instead of hardcoding credentials
    print("RTSP Video Stream Test")
    print("=" * 30)

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

            snapshot_count = 0
            print("Taking snapshots every 1 second. Press Ctrl+C to stop...")

            while True:
                try:
                    # Take snapshot with custom directory structure
                    file_path = stream.save_snapshot()

                    if file_path:
                        snapshot_count += 1
                        print(f"Snapshot #{snapshot_count}: {file_path}")

                    # Check stream status periodically
                    if snapshot_count % 10 == 0:  # Every 10 snapshots
                        status = stream.get_status()
                        print(f"Stream status: {status}")

                    time.sleep(1)

                except ValueError:
                    print("No frame available, waiting...")
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
