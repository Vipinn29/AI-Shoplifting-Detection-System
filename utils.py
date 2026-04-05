"""
Utility functions for the shoplifting detection system.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a video frame for model inference.

    Args:
        frame: Input frame as numpy array (H, W, C)
        target_size: Target size for the model input (height, width)

    Returns:
        Preprocessed frame
    """
    # Resize frame
    resized = cv2.resize(frame, target_size)

    # Convert BGR to RGB if needed (depending on model)
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] or mean/std normalize depending on model
    # For now, just return resized frame
    return resized


def draw_bounding_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box on the frame.

    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: Box color in BGR
        thickness: Line thickness

    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_text_on_frame(frame: np.ndarray, text: str, color: Tuple[int, int, int] = (0, 255, 0),
                       position: Tuple[int, int] = (10, 30), font_scale: float = 0.7, thickness: int = 2) -> np.ndarray:
    """
    Draw text on a video frame with a small background for readability.

    Args:
        frame: Input frame
        text: Text to display
        color: Text color in BGR
        position: Text bottom-left position
        font_scale: Scale for text size
        thickness: Text thickness

    Returns:
        Frame with text drawn
    """
    overlay = frame.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(overlay, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), cv2.FILLED)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame


def calculate_moving_average(buffer):
    """
    Calculate the moving average of a numeric buffer.

    Args:
        buffer: Sequence of numeric values

    Returns:
        float: moving average (0 if buffer is empty)
    """
    if not buffer:
        return 0.0

    return float(sum(buffer) / len(buffer))