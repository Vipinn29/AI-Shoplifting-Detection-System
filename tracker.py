"""
Tracker module for person detection using YOLOv8.
"""

import logging
from typing import List, Tuple
import cv2

logger = logging.getLogger(__name__)


class PersonDetection:
    """Result of person detection."""

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"PersonDetection(bbox={self.bbox}, confidence={self.confidence:.2f})"


class PersonTracker:
    """YOLOv8-based person detector with motion tracking."""

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5, max_trail_length: int = 20):
        """
        Initialize the person tracker.

        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8n.pt', 'yolov8s.pt')
            conf_threshold: Confidence threshold for detections
            max_trail_length: Maximum number of positions to store in trail
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.max_trail_length = max_trail_length
        self.model = None
        self.track_history = []  # List of trails, each trail is a list of (x,y) center points
        self.load_yolo_model()

    def load_yolo_model(self) -> None:
        """Load the YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded YOLOv8 model: {self.model_name}")
        except ImportError:
            logger.error("ultralytics not installed. Please install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None

    def detect_persons(self, frame) -> List[PersonDetection]:
        if self.model is None:
            logger.warning("YOLO model not loaded. Returning empty detections.")
            return []
    
        try:
            results = self.model.predict(frame, conf=self.conf_threshold, classes=[0], verbose=False)
    
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
    
                        detections.append(PersonDetection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(confidence)
                        ))
    
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:3]
    
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return []

    def update_tracks(self, detections: List[PersonDetection]) -> None:
        """
        Update tracking trails based on current detections.

        Args:
            detections: List of PersonDetection objects
        """
        # For simplicity, create/update trails for each detection
        # In a more advanced implementation, this would match detections to existing tracks
        new_trails = []

        for det in detections:
            # Calculate ROI center from upper 40% (focus torso/hands, ignore legs)
            x1, y1, x2, y2 = det.bbox
            height = y2 - y1
            upper_height = int(height * 0.4)
            center_x = (x1 + x2) // 2
            center_y = y1 + (upper_height // 2)
            center = (center_x, center_y)

            # Find closest existing trail or create new one
            closest_trail_idx = None
            min_distance = float('inf')

            for i, trail in enumerate(self.track_history):
                if trail:  # Check if trail has points
                    last_point = trail[-1]
                    distance = ((center[0] - last_point[0]) ** 2 + (center[1] - last_point[1]) ** 2) ** 0.5
                    if distance < 50 and distance < min_distance:  # 50 pixel threshold
                        min_distance = distance
                        closest_trail_idx = i

            if closest_trail_idx is not None:
                # Update existing trail
                trail = self.track_history[closest_trail_idx]
                trail.append(center)
                if len(trail) > self.max_trail_length:
                    trail.pop(0)  # Remove oldest point
                new_trails.append(trail)
            else:
                # Create new trail
                new_trails.append([center])

        # Update track history, keeping only active trails
        self.track_history = new_trails

    def draw_detections(self, frame, detections: List[PersonDetection]) -> None:
        """
        Draw bounding boxes and labels for detected persons.

        Args:
            frame: Input frame to draw on
            detections: List of PersonDetection objects
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box

            # Draw label
            label = f"Person: {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def draw_tracking_trail(self, frame, track_history: List[List[Tuple[int, int]]]) -> None:
        """
        Draw motion tracking trails for detected persons.

        Args:
            frame: Input frame to draw on
            track_history: List of trails, each trail is a list of (x,y) center points
        """
        for trail in track_history:
            if len(trail) < 2:
                continue  # Need at least 2 points to draw a line

            # Draw lines connecting the trail points
            for i in range(1, len(trail)):
                # Draw line from previous point to current point
                cv2.line(frame, trail[i-1], trail[i], (255, 0, 0), 2)  # Blue color, thickness 2

                # Optionally draw small circles at each point for better visibility
                cv2.circle(frame, trail[i], 3, (255, 0, 0), -1)  # Blue filled circle