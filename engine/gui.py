import time
from collections import deque
from typing import Dict

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


class GUIUtils:
    def __init__(self, fps_window=1.0):
        if fps_window <= 0:
            raise ValueError("fps_window must be positive")

        self.fps_window = fps_window  # Time window in seconds for FPS calculation
        self.frame_times = deque()
        self.fps = 0

    def show_fps(self, frame: np.ndarray) -> np.ndarray:
        """
        Display FPS counter on the given frame.

        Args:
            frame: OpenCV frame (numpy array) to draw FPS on

        Returns:
            Modified frame with FPS counter overlay
        """
        # Record current frame time
        current_time = time.time()
        self.frame_times.append(current_time)

        # Remove timestamps outside the time window
        while self.frame_times and current_time - self.frame_times[0] > self.fps_window:
            self.frame_times.popleft()

        # Calculate average FPS over the time window
        if len(self.frame_times) > 1:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.fps = (len(self.frame_times) - 1) / time_span
            else:
                self.fps = 0
        else:
            self.fps = 0

        # Create FPS text
        fps_text = f"FPS: {self.fps:.1f}"

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, font, font_scale, thickness
        )

        # Position for top-left corner with some padding
        x, y = 10, 30

        # Draw black background rectangle
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1,
        )

        # Draw white text on top
        cv2.putText(frame, fps_text, (x, y), font, font_scale, text_color, thickness)

        return frame

    def draw_detection_cv2(
        self,
        frame: np.ndarray,
        bboxes: list,
        class_ids: list,
        scores: list,
        class_names: Dict[int, str],
        line_width: int = 2,
    ):

        # Draw detections on the frame
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            # Get class name
            class_name = class_names[class_id]

            # Label
            label_text = f"{class_name}: {int(score * 100)}%"

            # Get color for this class
            color = colors(class_id, True)

            # Draw bounding box
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, line_width)

            labelSize, baseLine = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_ymin = max(ymin, labelSize[1] + 10)

            # Draw label background
            cv2.rectangle(
                frame,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                color,
                cv2.FILLED,
            )

            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text on colored background
                1,
            )
        return frame

    def draw_detection_annotator(
        self,
        frame: np.ndarray,
        bboxes: list,
        class_ids: list,
        scores: list,
        class_names: Dict[int, str],
        line_width: int = 2,
    ):
        annotator = Annotator(frame, line_width=line_width)

        # Draw detections on the frame
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            # Get class name
            class_name = class_names[class_id]

            # Label
            label_text = f"{class_name}: {int(score * 100)}%"

            # Draw bounding box
            annotator.box_label(bbox, label_text, color=colors(class_id, True))

        return annotator.result()
