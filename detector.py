"""
Detector module for loading models and performing inference.
"""

import os
import logging
from typing import List, Tuple, Optional, Any
import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class DetectionResult:
    """Result of a detection inference."""

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.smoothed_prob = 0.0
        self.alpha = 0.2

        self.last_active_time = 0
        self.active_start_time = None

        self.state = "NORMAL"

    def __repr__(self) -> str:
        return f"DetectionResult(bbox={self.bbox}, confidence={self.confidence:.2f}, class={self.class_name})"


class Detector:
    """Base detector class for loading models and performing inference."""

    def __init__(self, model_path: str):
        """
        Initialize the detector.

        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.load_model()
        self.smoothed_prob = 0.0
        self.alpha = 0.2
        
        self.last_active_time = 0
        self.active_start_time = None
        
        self.state = "NORMAL"
        
    def load_model(self) -> None:
        """Load the Keras .h5 model."""
        if not os.path.exists(self.model_path):
            logger.info(f"Custom model '{self.model_path}' not found. Rule-based detection active.")
            self.model = None
        else:
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            except ImportError:
                logger.error("TensorFlow not installed. Please install TensorFlow to use model inference.")
                self.model = None
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a video frame for model inference.

        Args:
            frame: Input frame as numpy array (H, W, C)

        Returns:
            Preprocessed frame ready for model input
        """
        # Resize to 224x224
        resized = cv2.resize(frame, (224, 224))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def predict(self, frame: np.ndarray) -> float:
        """
        Predict shoplifting probability from a frame.

        Args:
            frame: Input frame

        Returns:
            Probability score (0.0 to 1.0)
        """
        if self.model is None:
            # Rule-based fallback (no random)
            return 0.0

        try:
            preprocessed = self.preprocess_frame(frame)
            predictions = self.model.predict(preprocessed, verbose=0)
            # Assuming binary classification, return probability of positive class
            probability = float(predictions[0][0]) if predictions.shape[-1] == 1 else float(predictions[0][1])
            return probability
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0.0

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:

        import time

        probability = self.predict(frame)

        # ---- SMOOTHING ----
        self.smoothed_prob = (
            self.alpha * probability +
            (1 - self.alpha) * self.smoothed_prob
        )

        current_time = time.time()

        # ---- NOISE FILTER ----
        if self.smoothed_prob < 0.2:
            self.smoothed_prob = 0

        # ---- ACTIVE LOGIC ----
        if self.smoothed_prob > 0.4:
            self.last_active_time = current_time

            if self.active_start_time is None:
                self.active_start_time = current_time

            self.state = "ACTIVE"

        # ---- HOLD ACTIVE (IMPORTANT) ----
        elif current_time - self.last_active_time < 1.5:
            self.state = "ACTIVE"

        else:
            self.state = "NORMAL"
            self.active_start_time = None

        # ---- SUSPICIOUS LOGIC ----
        if self.active_start_time is not None:
            if current_time - self.active_start_time > 3:
                self.state = "SUSPICIOUS"

        # ---- RETURN DETECTION ONLY IF SUSPICIOUS ----
        if self.state == "SUSPICIOUS":
            h, w = frame.shape[:2]
            return [DetectionResult(
                bbox=(0, 0, w, h),
                confidence=self.smoothed_prob,
                class_id=1,
                class_name="shoplifting"
            )]

        return []

class ShopliftingDetector(Detector):
    """Specialized detector for shoplifting detection."""

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Override to add shoplifting-specific logic."""
        detections = super().detect(frame)

        # Additional logic can be added here if needed
        # For example, temporal smoothing, multiple frame analysis, etc.

        return detections