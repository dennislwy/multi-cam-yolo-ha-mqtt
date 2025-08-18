from typing import Tuple

import cv2
import numpy as np


def calculate_meters_per_pixel(p1, p2, real_distance):
    """
    Calculate the number of meters per pixel between two points.

    Args:
        p1 (tuple): Coordinates of the first point (x1, y1).
        p2 (tuple): Coordinates of the second point (x2, y2).
        real_distance (float): The real-world distance between the two points in meters.

    Returns:
        float: The number of meters per pixel.
    """
    # Calculate the Euclidean distance between the two points
    distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    if distance == 0:
        raise ValueError("The two points must not be the same.")

    # Calculate meters per pixel
    meters_per_pixel = real_distance / distance
    return meters_per_pixel


def get_wrap_matrix(roi: np.ndarray, dst_size: Tuple[int, int]):
    """
    Get the wrap matrix for perspective transformation.

    Args:
        roi (np.ndarray): Region of interest defined by four points.
        dst_size (Tuple[int, int]): Size of the destination image (width, height).
    """
    if roi.shape[0] != 4 or roi.shape[1] != 2:
        raise ValueError("ROI must contain exactly four points.")

    # Define the destination points for the perspective transformation
    dst_points = np.array(
        [[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]],
        dtype=np.float32,
    )

    # Calculate the wrap matrix using cv2.getPerspectiveTransform
    wrap_matrix = cv2.getPerspectiveTransform(roi.astype(np.float32), dst_points)
    return wrap_matrix


def warp_perspective(frame, warp_matrix: np.ndarray, dst_size: Tuple[int, int]):
    """
    Apply a perspective warp to the input frame.

    Args:
        frame (np.ndarray): The input image frame.
        warp_matrix (np.ndarray): The transformation matrix.
        dst_size (Tuple[int, int]): Size of the destination image (width, height).

    Returns:
        np.ndarray: The warped image.
    """
    warped_frame = cv2.warpPerspective(frame, warp_matrix, dst_size)
    return warped_frame
