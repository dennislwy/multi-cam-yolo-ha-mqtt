from pathlib import Path
from typing import Union

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ObjectDetection:
    def __init__(self, weights_path: Union[str, Path], device=None):
        """
        Initialize the ObjectDetection class.

        Args:
            weights_path: Path to the YOLO model weights file
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model = YOLO(weights_path, task="detect")
        self.device = self._parse_device(device)
        print(f"Using device: {self.device}")
        # self.model.to(self.device)

        # Get class names from the model
        self.class_names = self.model.names

        self._last_class_ids = []

    def get_color(self, class_id):
        return colors(class_id, True)

    def detect(self, frame, imgsz=640, conf=0.25, nms=0.45, classes=None, device=None):
        """
        Run object detection on a single frame.

        Args:
            frame: The input image frame.
            imgsz: The size to resize the image.
            conf: The confidence threshold for detections.
            nms: The non-maximum suppression threshold.
            classes: The list of classes to detect (None for all classes).
            device: The device to run the model on (uses init device if None).

        Returns:
            Tuple of (bboxes, class_ids, scores)
            - bboxes: List of [xmin, ymin, xmax, ymax] coordinates
            - class_ids: List of class indices
            - scores: List of confidence scores
        """
        # Use the initialized device if none is specified
        if device is None:
            device = self.device

        # Run inference
        results = self.model(
            frame,
            imgsz=imgsz,
            conf=conf,
            iou=nms,
            classes=classes,
            device=device,
            verbose=False,
        )

        # Extract results from the first (and only) image
        result = results[0]

        bboxes = []
        scores = []
        class_ids = []  # Keep track of class IDs for color indexing

        if result.boxes is not None:
            # Get bounding boxes in xyxy format
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf_score, class_id in zip(boxes, confidences, class_indices):
                xmin, ymin, xmax, ymax = box.astype(int)

                bboxes.append([xmin, ymin, xmax, ymax])
                scores.append(conf_score)
                class_ids.append(class_id)

        # Store class_ids for color access in the main loop
        self._last_class_ids = class_ids

        return (bboxes, class_ids, scores)

    @property
    def classes(self):
        """
        Property to access class names as a dictionary, e.g. od.classes[class_id]
        """
        return self.class_names

    def _parse_device(self, device: str = None):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"CUDA detected. Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                device = "cpu"
                print("CUDA not available. Using CPU for inference.")
        else:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print("CUDA not available. Falling back to CPU.")
                    device = "cpu"
                elif device != "cuda" and ":" in device:
                    try:
                        gpu_idx = int(device.split(":")[1])
                        if gpu_idx >= torch.cuda.device_count():
                            print(
                                f"GPU {gpu_idx} not available. Using default CUDA device."
                            )
                            device = "cuda"
                        else:
                            # Show specific GPU info
                            gpu_name = torch.cuda.get_device_name(gpu_idx)
                            gpu_memory = (
                                torch.cuda.get_device_properties(gpu_idx).total_memory
                                / 1024**3
                            )
                            print(
                                f"Using GPU {gpu_idx}: {gpu_name} ({gpu_memory:.1f} GB)"
                            )
                    except (ValueError, IndexError):
                        print("Invalid CUDA device format. Using default CUDA device.")
                        device = "cuda"
                else:
                    # Show default GPU info
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return device
