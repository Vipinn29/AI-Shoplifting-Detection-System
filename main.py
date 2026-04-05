"""
Main entry point for the real-time shoplifting detection system.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "3"
import sys
import warnings
warnings.filterwarnings("ignore")
import time
import signal
import logging
import threading
import cv2
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple
from queue import Queue
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torch import dist
import config
from alert import TelegramAlert, send_alert
from utils import draw_bounding_box
from tracker import PersonTracker, PersonDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ShopliftingDetectionSystem:
    """Main system class for shoplifting detection."""

    def __init__(self):
        self.alert_system = TelegramAlert()
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.score_buffer = deque(maxlen=config.FRAME_BUFFER)
        self.alert_state = False
        self.alert_folder = Path(os.path.join(os.path.dirname(__file__), 'alerts'))
        self.alert_folder.mkdir(parents=True, exist_ok=True)
        self.tracker = PersonTracker()
        self.last_alert_time = 0  # Timestamp of last alert message
        
        # Alert locking mechanism
        self.alert_active = False  # Flag to prevent multiple alerts for same event
        self.alert_time = 0  # Timestamp when alert was triggered
        
        self.frame_times = []  # For FPS calculation
        self.frame_counter = 0  # Frame counter for skipping frames
        self.detection_interval = 10  # Process detection every 10 frames
        self.last_person_detections = []  # Cache for detections
        self.last_person_probabilities = []  # Cache for probabilities
        self.person_motion_history = {}  # Track center positions per person: {person_id: [(x,y), ...]}
        self.motion_history_length = 30  # Keep 30 frames of motion history for idle detection
        self.idle_threshold = 10  # Pixels - movement below this = idle
        
        # Rule-based suspicious detection data structures
        self.person_bbox_history = {}  # Track bounding box dimensions: {person_id: [(w,h), ...]}
        self.person_movement_states = {}  # Track movement state per person: {person_id: 'idle'|'active'}
        self.person_continuous_movement = {}  # Track continuous movement frames: {person_id: count}
        self.suspicious_threshold = 30  # Frames of continuous movement to trigger suspicious
        self.bbox_change_threshold = 50  # Pixel change threshold for sudden size changes
        
        # Enhanced behavior-based detection data structures
        self.person_behavior_history = {}  # Track behavior patterns: {person_id: {'bending': count, 'fast_movement': count, 'restless': count}}
        self.fast_movement_threshold = 15  # Pixels per frame for fast movement
        self.height_change_threshold = 30  # Pixels for significant height decrease (bending)
        self.restless_threshold = 8  # Number of direction changes for restless behavior
        self.behavior_history_length = 20  # Frames to track behavior patterns
        
        # Hand/upper-body movement tracking for shoplifting detection
        self.person_upper_body_history = {}  # Track upper-half center positions: {person_id: [(x,y), ...]}
        self.person_upper_body_movement_intensity = {}  # Track movement intensity in upper region: {person_id: intensity_score}
        self.upper_body_movement_threshold = 3  # Reduced for subtle hand motion
        self.upper_body_history_length = 15  # Frames to track upper-body movement
        
        # Stability logic data structures
        self.person_status_history = {}  # Track status history: {person_id: [status1, status2, ...]}
        self.status_history_length = 20  # Keep 20 frames of status history
        self.suspicious_stability_threshold = 10  # Need 10+ suspicious frames in last 20 to mark as suspicious
        self.person_active_frame_counts = {}  # Track continuous ACTIVE frames per person
        self.person_suspicious_frame_counts = {}  # Track continuous SUSPICIOUS frames for alert escalation
        
        # State memory for preventing rapid state switching
        self.person_committed_status = {}  # Committed state that doesn't downgrade immediately: {person_id: status}
        self.person_normal_frame_counts = {}  # Count normal frames when in SUSPICIOUS state: {person_id: count}
        
        # Visualization enhancements
        self.person_id_counter = 0  # Counter for assigning numeric IDs
        self.person_id_mapping = {}  # Map string IDs to numeric display IDs: {string_id: numeric_id}
        self.person_state_timers = {}  # Track frames in current state: {person_id: frame_count}
        self.person_last_status = {}  # Track last known status to detect changes: {person_id: status}
        
        # Visualization enhancements
        self.person_id_counter = 0  # Counter for assigning numeric IDs
        self.person_id_mapping = {}  # Map string IDs to numeric display IDs: {string_id: numeric_id}
        self.person_state_timers = {}  # Track frames in current state: {person_id: frame_count}
        self.person_last_status = {}  # Track last known status to detect changes: {person_id: status}
        
        # Alert visualization
        self.has_current_alert = False  # Flag for current frame alert state
        self.alert_flash_frame = 0  # Frame counter for flashing effect
        
        # Motion trail tracking (list of center points for each person)
        self.motion_trails = {}  # person_id -> list of (x, y) points
        self.max_trail_length = 15  # Maximum number of points in trail
        
        # Motion filtering constants (relaxed range for real hand movement)
        self.MIN_MOTION_AREA = 80   # Further reduced, accept tiny motions
        self.MAX_MOTION_AREA = 3000 # Increased range, accept larger hand motion
        self.DIFF_THRESHOLD = 20  # Keep sensitivity
        
        # Debug overlay
        self.debug_motion_score = 0.0
        self.debug_motion_area = 0
        self.debug_person_id = None
        self.previous_alert_state = None  # Track previous alert state to prevent spam
        self.suspicious_frames_count = 0  # Consecutive SUSPICIOUS frames counter
        
        # Per-person prev ROI for diff
        self.person_prev_roi = {}  # person_id -> prev_roi
        
        # Idle skip counter
        self.person_state_hold_counter = {}  # person_id -> ACTIVE hold frames remaining
        self.motion_interval = 2  # Motion analysis every 2 frames
        
        # Duration-based suspicious detection
        self.person_active_start_times = {}  # {person_id: start_time} for 3s ACTIVE timer
        self.person_motion_states = {}
        self.person_last_motion_times = {}
        

        # EMA smoothing for motion stability
        self.smoothed_motion_scores = {}  # person_id -> smoothed_motion (0.0-1.0)
        
        # Hysteresis + time-hold for state stability
        self.active_enter_threshold = 0.003
        self.active_exit_threshold = 0.0015
        self.active_hold_duration = 1.5
        self.person_last_motion_times = {}  # person_id -> last time above ENTER
        
        # Threading components

        self.frame_queue = Queue(maxsize=10)  # Queue for frames between capture and detection threads
        self.capture_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.display_frame: Optional[cv2.Mat] = None  # Latest processed frame for display
        self.frame_lock = threading.Lock()  # Lock for thread-safe access to display_frame
        self.detection_lock = threading.Lock()  # Lock for thread-safe access to detection results
        self.prev_frame = None  # For motion detection in capture thread
        self.prev_gray = None
        self.smoothed_motion_scores = {}

        self.person_motion_states = {}
        self.person_last_motion_times = {}

        self.motion_history = {}  # person_id -> list of recent motion values

        self.person_motion_counter = {}
        self.motion_points = []
        self.person_hand_active = {}
        self.person_last_hand_time = {}
        self.person_active_start_times = {}

        self.hand_trails = {}   # hand_0 -> [(x,y), ...], hand_1 -> [(x,y), ...]

        self.person_motion_streak = {}
        self.person_valid_zone = {}

        # Hand detection disabled - model download issue
        mp_hands = mp.solutions.hands

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_hand_positions = {}
        self.person_hand_active = {}   # 🔥 NEW


    def _camera_capture_thread(self) -> None:
       """Thread for continuous camera frame capture."""
       logger.info("Camera capture thread started")

       while self.running:
           try:
               ret, frame = self.cap.read()
               if not ret:
                   logger.warning("Failed to read frame from video source")
                   time.sleep(0.05)
                   continue
               
               # ---------------- FRAME RESIZE + QUEUE ----------------
               frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

               # ---------------- HAND DETECTION ----------------
               rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               results = self.hands.process(rgb_frame)

               # reset hand motion points every frame
               self.motion_points = []

               if results.multi_hand_landmarks:
                   for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                       wrist = hand_landmarks.landmark[0]

                       frame_h, frame_w, _ = frame.shape
                       cx = int(wrist.x * frame_w)
                       cy = int(wrist.y * frame_h)

                       prev_key = f"hand_{idx}"

                       if prev_key not in self.hand_trails:
                            self.hand_trails[prev_key] = []

                       self.hand_trails[prev_key].append((cx, cy))
                       if len(self.hand_trails[prev_key]) > 20:
                           self.hand_trails[prev_key].pop(0)

                       if prev_key in self.prev_hand_positions:
                            px, py = self.prev_hand_positions[prev_key]
                            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

                            # ignore tiny noise
                            if dist < 4:
                                continue

                            # only consider significant hand movement to reduce noise
                            if dist > 10:  # Reduced threshold for subtle hand motion
                                hand_y_ratio = cy / frame_h

                                # stricter lower-body zone
                                if 0.60 < hand_y_ratio < 0.96: # Focus on hands in lower body zone
                                    self.motion_points.append((cx, cy))

                       self.prev_hand_positions[prev_key] = (cx, cy)
               else:
                   self.prev_hand_positions.clear()
                   self.hand_trails.clear()

               try:
                   self.frame_queue.put(frame, block=False)
               except:
                   pass

           except Exception as e:
               logger.error(f"Error in camera capture thread: {e}")
               time.sleep(0.05)

       logger.info("Camera capture thread stopped")

    def _draw_hand_trail(self, frame: cv2.Mat) -> None:
       if not hasattr(self, "hand_trails"):
           return

       colors = [
           (0, 255, 255),   # yellow
           (255, 255, 0),   # cyan
       ]

       for idx, (hand_key, trail) in enumerate(self.hand_trails.items()):
           if len(trail) < 2:
               continue

           color = colors[idx % len(colors)]

           for i in range(1, len(trail)):
               pt1 = trail[i - 1]
               pt2 = trail[i]
               alpha = i / len(trail)
               thickness = max(1, int(1 + alpha * 2))
               trail_color = tuple(int(c * alpha) for c in color)

               cv2.line(frame, pt1, pt2, trail_color, thickness)

           cv2.circle(frame, trail[-1], 5, color, -1)

    def _detection_processing_thread(self) -> None:
        """Thread for YOLO detection and frame processing."""
        logger.info("Detection processing thread started")
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except:
                    # Timeout, continue loop
                    continue
                
                # Process the frame
                processed_frame = self._process_frame_detection(frame)
                
                # Update display frame with thread safety
                with self.frame_lock:
                    self.display_frame = processed_frame.copy()
                
                self.frame_counter += 1
                
            except Exception as e:
                logger.error(f"Error in detection processing thread: {e}")
        
        logger.info("Detection processing thread stopped")

    def initialize(self) -> bool:
        """Initialize the video capture."""
        try:
            self.cap = cv2.VideoCapture(config.CAMERA_ID)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {config.CAMERA_ID}")
                return False

            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)

            logger.info("Video capture initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            return False

    def _save_frame(self, frame: cv2.Mat) -> Path:
        """Save a frame image in alert directory."""
        timestamp = int(time.time() * 1000)
        output_path = self.alert_folder / f"alert_{timestamp}.jpg"
        if cv2.imwrite(str(output_path), frame):
            logger.info(f"Saved alert snapshot to {output_path}")
        else:
            logger.error(f"Failed to save alert snapshot to {output_path}")
        return output_path

    def _trigger_alert(self, frame, avg_score):

        def async_alert():
            image_path = self._save_frame(frame)
            current_time = time.strftime("%H:%M:%S")

            message = (
                "🚨 Suspicious Activity Detected\n"
                f"Time: {current_time}\n"
                "Location: Waist/Pocket Zone\n"
                "Camera: Cam 1"
            )

            # self.alert_system._send_alert([], frame=frame)
            send_alert(message, str(image_path))

        threading.Thread(target=async_alert, daemon=True).start()


    def _draw_person_detections(self, frame: cv2.Mat, detections: List[PersonDetection], probabilities: List[float]) -> None:
        """
        Draw person detections with enhanced visualization.

        Args:
            frame: Frame to draw on
            detections: List of PersonDetection objects
            probabilities: List of probabilities corresponding to detections (for compatibility)
        """
        for detection, probability in zip(detections, probabilities):
            x1, y1, x2, y2 = detection.bbox
            person_id = self._get_person_id(detection.bbox)
            display_id = self._get_display_id(person_id)

            # Get stable status from rule-based detection (with stability logic)
            status = self._get_stable_person_status(person_id)
            
            # Track state timer for display
            last_status = self.person_last_status.get(person_id)
            if last_status != status:
                # Status changed, reset timer
                self.person_state_timers[person_id] = 1
                self.person_last_status[person_id] = status
            else:
                # Same status, increment timer
                self.person_state_timers[person_id] = self.person_state_timers.get(person_id, 0) + 1
            
            state_timer = self.person_state_timers.get(person_id, 0)
            
            # Set color based on status (confirmed: NORMAL=Green, ACTIVE=Yellow, SUSPICIOUS=Red)
            if status == "NORMAL":
                color = (0, 255, 0)  # Green
                bg_color = (0, 255, 0)
                box_thickness = 2
            elif status == "ACTIVE":
                color = (0, 255, 255)  # Yellow
                bg_color = (0, 255, 255)
                box_thickness = 3
            elif status == "SUSPICIOUS":
                color = (0, 0, 255)  # Red
                bg_color = (0, 0, 255)
                box_thickness = 4
            elif status == "ALERT":
                color = (0, 0, 255)  # Red
                bg_color = (0, 0, 255)
                box_thickness = 5
                self.has_current_alert = True
            elif status == "IDLE":
                color = (255, 0, 0)  # Blue
                bg_color = (255, 0, 0)
                box_thickness = 2
            else:
                color = (128, 128, 128)  # Gray
                bg_color = (128, 128, 128)
                box_thickness = 2

            # Minimal corner box
            corner_len = 30
            th = box_thickness

            # top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, th)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, th)

            # top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, th)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, th)

            # bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, th)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, th)

            # bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, th)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, th)

            # Draw upper body motion arrow (professional visualization)
            if len(self.person_upper_body_history.get(person_id, [])) >= 2:
                upper_centers = self.person_upper_body_history[person_id]
                prev_center = upper_centers[-2]
                curr_center = upper_centers[-1]
                cv2.arrowedLine(frame, prev_center, curr_center, color, 2, tipLength=0.3)

            # Draw motion trail if available (full body)
            self._draw_motion_trail(frame, person_id, color)

            # Create label with timer: "ID X - STATE (timer)"
            if status == 'ALERT':
                label = f"ID {display_id} | ALERT"
            else:
                label = f"ID {display_id} | {status}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            pad = 8
            bg_x1 = x1
            bg_y1 = max(0, y1 - text_height - baseline - 14)
            bg_x2 = x1 + text_width + pad * 2
            bg_y2 = y1 - 4
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            
            cv2.putText(
                frame,
                label,
                (bg_x1 + pad, bg_y2 - 8),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    def _draw_motion_trail(self, frame: cv2.Mat, person_id: str, color: tuple) -> None:
        if person_id not in self.motion_trails:
            return
    
        trail = self.motion_trails[person_id]
        if len(trail) < 2:
            return
    
        for i in range(1, len(trail)):
            pt1 = trail[i - 1]
            pt2 = trail[i]
    
            # newer segments brighter and thicker
            alpha = i / len(trail)
            thickness = max(1, int(1 + alpha * 3))
            trail_color = tuple(int(c * alpha) for c in color)
    
            cv2.line(frame, pt1, pt2, trail_color, thickness)
    
        # latest point highlight
        last_pt = trail[-1]
        cv2.circle(frame, last_pt, 5, color, -1)

    def _draw_overlay(self, frame: cv2.Mat, person_count: int, max_probability: float) -> None:
        """
        Draw professional surveillance-style overlay.

        Args:
            frame: Frame to draw on
            person_count: Number of persons detected
            max_probability: Maximum probability across all persons
        """
        height, width = frame.shape[:2]

        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        # Keep only last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)

        fps = 0.0
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

        # Draw semi-transparent overlay panel at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # title
        cv2.putText(frame, "AI SHOPLIFTING SURVEILLANCE", (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        # stats
        cv2.putText(frame, f"Persons: {person_count}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        
        timestamp = time.strftime("%H:%M:%S", time.localtime(current_time))
        cv2.putText(frame, f"Time: {timestamp}", (260, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Status information
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1

        # system state
        if self.alert_active or max_probability >= 0.8:
            state_text = "ALERT"
            state_color = (0, 0, 255)
        elif max_probability >= 0.4:
            state_text = "MONITORING"
            state_color = (0, 255, 255)
        else:
            state_text = "NORMAL"
            state_color = (0, 255, 0)

        cv2.putText(frame, f"System: {state_text}", (frame.shape[1] - 180, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)

        # Debug motion overlay
        debug_font_scale = 0.5
        if self.debug_motion_score > 0:
            debug_text = f"Motion: {self.debug_motion_score:.2f} (ID:{self.debug_person_id[-4:] if self.debug_person_id else 'N/A'})"
            cv2.putText(frame, debug_text, (20, height - 60), font, debug_font_scale, (0, 255, 255), 1)
        
        # Status indicator
        # status_y = height - 30
        # if self.alert_active:
        #     status_text = "STATUS: ALERT ACTIVE"
        #     status_color = (0, 0, 255)
        # elif max_probability > 0.9:
        #     status_text = "STATUS: ALERT ACTIVE"
        #     status_color = (0, 0, 255)
        # elif max_probability > 0.6:
        #     status_text = "STATUS: MONITORING"
        #     status_color = (0, 255, 255)
        # else:
        #     status_text = "SYSTEM STATUS: NORMAL"
        #     status_color = (0, 255, 0)

        # # Draw status background
        # status_overlay = frame.copy()
        # cv2.rectangle(status_overlay, (10, status_y - 20), (width - 10, status_y + 5), (0, 0, 0), -1)
        # cv2.addWeighted(status_overlay, 0.7, frame, 0.3, 0, frame)

        # cv2.putText(frame, status_text, (20, status_y), font, 0.6, status_color, 2)

    def _draw_alert_border(self, frame: cv2.Mat) -> None:
       if not self.alert_active and not self.has_current_alert:
           return

       self.alert_flash_frame = (self.alert_flash_frame + 1) % 6
       if self.alert_flash_frame < 3:
           h, w = frame.shape[:2]
           color = (0, 0, 255)
           th = 6

           cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, th)

           cv2.putText(frame, "ALERT MODE", (w - 170, 28),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _calculate_weighted_average(self, values: List[float]) -> float:
        """
        Calculate weighted moving average with recent values weighted higher.

        Args:
            values: List of values to average

        Returns:
            Weighted average (higher weight on recent values)
        """
        if not values:
            return 0.0

        # Create weights: exponentially increasing towards the end
        weights = [2 ** (i / len(values)) for i in range(len(values))]
        total_weight = sum(weights)
        weighted_sum = sum(v * w for v, w in zip(values, weights))

        return weighted_sum / total_weight


    def _get_person_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Generate a unique ID for a person based on bounding box position.
        IDs are based on approximate location and trusted for 20 frames.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Person ID string
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # Quantize to grid for stability
        grid_size = 40
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size
        return f"person_{grid_x}_{grid_y}"

    def _get_display_id(self, person_id: str) -> int:
        """
        Get a numeric display ID for a person (for cleaner visualization).

        Args:
            person_id: Internal person ID string

        Returns:
            Numeric display ID
        """
        if person_id not in self.person_id_mapping:
            self.person_id_counter += 1
            self.person_id_mapping[person_id] = self.person_id_counter
        
        return self.person_id_mapping[person_id]

    def _update_person_motion(self, frame: cv2.Mat, person_detections: List[PersonDetection]) -> None:
        """
        Update motion history for each person by tracking their center positions and bounding box dimensions.
        Perform motion analysis ONLY on upper 40% ROI.

        Args:
            frame: Current frame
            person_detections: List of detected persons
        """
        for detection in person_detections:
            person_id = self._get_person_id(detection.bbox)
            
            # Crop upper 40% ROI for frame-diff motion detection
            curr_roi = self._crop_upper_region(frame, detection.bbox)
            
            # Ensure consistent size for frame-diff - FIXED 64x72 for speed
            if person_id in self.person_prev_roi and self.person_prev_roi[person_id] is not None:
                target_size = (64, 72)  # Small fixed size for fast processing
                prev_roi_resized = cv2.resize(self.person_prev_roi[person_id], target_size)
                curr_roi_resized = cv2.resize(curr_roi, target_size)
                diff = cv2.absdiff(prev_roi_resized, curr_roi_resized)
                motion_mask = diff > self.DIFF_THRESHOLD
                motion_area = np.sum(motion_mask)
                if self.MIN_MOTION_AREA <= motion_area <= self.MAX_MOTION_AREA:
                    # Meaningful hand motion - proceed
                    pass
                else:
                    # Ignore noise/large changes
                    motion_area = 0
            else:
                motion_area = 0
            
            # Store current ROI for next frame
            self.person_prev_roi[person_id] = curr_roi.copy()
            
            # Debug: track highest motion (EMA smoothed)
            current_motion = motion_area / (curr_roi.shape[0] * curr_roi.shape[1])  # Normalized 0-1
            
            # EMA smoothing: alpha=0.2, init=0
            if person_id not in self.smoothed_motion_scores:
                self.smoothed_motion_scores[person_id] = 0.0
            alpha = 0.2
            self.smoothed_motion_scores[person_id] = (alpha * current_motion) + ((1 - alpha) * self.smoothed_motion_scores[person_id])
            if self.smoothed_motion_scores[person_id] < 0.08:
                self.smoothed_motion_scores[person_id] = 0.0
            
            if self.smoothed_motion_scores[person_id] > self.debug_motion_score:
                self.debug_motion_score = self.smoothed_motion_scores[person_id]
                self.debug_motion_area = motion_area
                self.debug_person_id = person_id
            
            # Highlight moving region for debug
            x1_roi, y1_roi, x2_roi, y2_roi = detection.bbox
            if motion_area > self.MIN_MOTION_AREA:
                roi_y2 = y1_roi + int((y2_roi - y1_roi) * 0.4)
                cv2.rectangle(frame, (x1_roi, y1_roi), (x2_roi, roi_y2), (0, 255, 255), 2)  # Yellow ROI box
            
            # Calculate ROI center
            x1, y1, x2, y2 = detection.bbox
            roi_center = self._get_upper_body_center(detection.bbox)
            center = roi_center if motion_area > 0 else roi_center  # Keep center, but filter scores below

            # Update motion trail for visualization
            if person_id not in self.motion_trails:
                self.motion_trails[person_id] = []
            self.motion_trails[person_id].append(center)
            # Keep trail length limited
            if len(self.motion_trails[person_id]) > self.max_trail_length:
                self.motion_trails[person_id].pop(0)
            
            # Calculate bounding box dimensions
            width = x2 - x1
            height = y2 - y1
            bbox_size = (width, height)
            
            # Initialize motion history if new person
            if person_id not in self.person_motion_history:
                self.person_motion_history[person_id] = []
                self.person_bbox_history[person_id] = []
                self.person_movement_states[person_id] = 'idle'
                self.person_continuous_movement[person_id] = 0
                self.person_behavior_history[person_id] = {
                    'bending': 0,
                    'fast_movement': 0,
                    'restless': 0,
                    'hand_movement': 0,
                    'last_direction': None,
                    'direction_changes': 0
                }
                self.person_status_history[person_id] = []
                self.person_upper_body_history[person_id] = []
                self.person_upper_body_movement_intensity[person_id] = 0
            
            # Add current position to history
            motion_history = self.person_motion_history[person_id]
            motion_history.append(center)
            
            # Add current bbox size to history
            bbox_history = self.person_bbox_history[person_id]
            bbox_history.append(bbox_size)
            
            # Track upper-body movement ONLY from cropped ROI (focus hands)
            upper_roi = self._crop_upper_region(frame, detection.bbox)  # Note: frame not passed? Wait, no frame in args - use bbox center for consistency
            upper_body_history = self.person_upper_body_history[person_id]
            upper_body_history.append(center)  # Use ROI center
            if len(upper_body_history) > self.upper_body_history_length:
                upper_body_history.pop(0)
            
            # Analyze behavior patterns
            self._analyze_behavior_patterns(person_id, center, bbox_size)
            
            # Analyze upper-body movement for hand/arm activity
            self._analyze_upper_body_movement(person_id, detection.bbox)
            
            # Keep only last N positions
            if len(motion_history) > self.motion_history_length:
                motion_history.pop(0)
            if len(bbox_history) > self.motion_history_length:
                bbox_history.pop(0)


        # Clean up motion trails, smoothed motion, states, times for persons no longer detected
        current_person_ids = {self._get_person_id(detection.bbox) for detection in person_detections}
        trails_to_remove = [pid for pid in self.motion_trails.keys() if pid not in current_person_ids]
        smoothed_to_remove = [pid for pid in self.smoothed_motion_scores.keys() if pid not in current_person_ids]
        states_to_remove = [pid for pid in self.person_motion_states.keys() if pid not in current_person_ids]
        times_to_remove = [pid for pid in self.person_last_motion_times.keys() if pid not in current_person_ids]
        active_times_to_remove = [pid for pid in self.person_active_start_times.keys() if pid not in current_person_ids]
        for pid in trails_to_remove:
            del self.motion_trails[pid]
        for pid in smoothed_to_remove:
            del self.smoothed_motion_scores[pid]
        for pid in states_to_remove:
            del self.person_motion_states[pid]
        for pid in times_to_remove:
            del self.person_last_motion_times[pid]
        for pid in active_times_to_remove:
            if pid in self.person_active_start_times:
                del self.person_active_start_times[pid]


    def _get_upper_body_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate center of upper half of bounding box (head/shoulders/hands region).

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Center point of upper half (x, y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        # Upper 40% region center (torso + hands, per task)
        height = y2 - y1
        upper_height = int(height * 0.4)
        upper_y = y1 + (upper_height // 2)
        return (center_x, upper_y)

    def _crop_upper_region(self, frame: cv2.Mat, bbox: Tuple[int, int, int, int]) -> cv2.Mat:
        """
        Extract ONLY upper 40% ROI (torso + hands) from frame.
        Ignore lower body completely as per task.

        Args:
            frame: Input frame
            bbox: Person bbox (x1,y1,x2,y2)

        Returns:
            Cropped ROI frame for motion analysis
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        y2_crop = y1 + int(height * 0.4)
        return frame[y1:y2_crop, x1:x2]

    def _analyze_upper_body_movement(self, person_id: str, bbox: Tuple[int, int, int, int]) -> None:
        """
        Analyze hand/upper-body movement intensity for shoplifting detection.

        Args:
            person_id: Person identifier
            bbox: Current bounding box
        """

        upper_body_history = self.person_upper_body_history.get(person_id, [])
        
        if len(upper_body_history) < 2:
            return
        
        # Calculate movement distance in upper-body region
        prev_x, prev_y = upper_body_history[-2]
        curr_x, curr_y = upper_body_history[-1]
        
        movement_distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
        
        # Update hand movement intensity
        behavior = self.person_behavior_history.get(person_id, {})
        
        # Detect frequent upper-body movement (hands reaching/handling items)
        if movement_distance > self.upper_body_movement_threshold:
            behavior['hand_movement'] = behavior.get('hand_movement', 0) + 1
        else:
            # Decay hand movement counter when person settles
            behavior['hand_movement'] = max(0, behavior.get('hand_movement', 0) - 1)
        
        # Cap the counter
        behavior['hand_movement'] = min(behavior['hand_movement'], self.behavior_history_length)

    def _analyze_behavior_patterns(self, person_id: str, current_center: Tuple[int, int], current_bbox: Tuple[int, int]) -> None:
        """
        Analyze behavior patterns for suspicious activity detection.

        Args:
            person_id: Person identifier
            current_center: Current center position (x, y)
            current_bbox: Current bounding box size (width, height)
        """
        behavior = self.person_behavior_history[person_id]
        motion_history = self.person_motion_history[person_id]
        bbox_history = self.person_bbox_history[person_id]
        
        # Need at least 2 frames for analysis
        if len(motion_history) < 2 or len(bbox_history) < 2:
            return
        
        # 1. Detect bending (sudden height decrease)
        prev_width, prev_height = bbox_history[-2]
        curr_width, curr_height = current_bbox
        
        height_decrease = prev_height - curr_height
        if height_decrease > self.height_change_threshold:
            behavior['bending'] += 1
            # Decay old bending detections
            behavior['bending'] = min(behavior['bending'], self.behavior_history_length)
        
        # 2. Detect fast movement (rapid center position changes)
        prev_x, prev_y = motion_history[-2]
        curr_x, curr_y = current_center
        
        movement_distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
        if movement_distance > self.fast_movement_threshold:
            behavior['fast_movement'] += 1
            # Decay old fast movement detections
            behavior['fast_movement'] = min(behavior['fast_movement'], self.behavior_history_length)
        
        # 3. Detect restless behavior (frequent direction changes)
        if len(motion_history) >= 3:
            # Calculate direction vectors
            prev_prev_x, prev_prev_y = motion_history[-3]
            prev_x, prev_y = motion_history[-2]
            curr_x, curr_y = current_center
            
            # Direction vectors
            dir1_x = prev_x - prev_prev_x
            dir1_y = prev_y - prev_prev_y
            dir2_x = curr_x - prev_x
            dir2_y = curr_y - prev_y
            
            # Calculate direction change (dot product)
            dot_product = dir1_x * dir2_x + dir1_y * dir2_y
            mag1 = (dir1_x ** 2 + dir1_y ** 2) ** 0.5
            mag2 = (dir2_x ** 2 + dir2_y ** 2) ** 0.5
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                # If angle > 90 degrees (significant direction change)
                if cos_angle < 0:
                    behavior['direction_changes'] += 1
                    
                    # Reset direction change counter periodically
                    if behavior['direction_changes'] >= self.restless_threshold:
                        behavior['restless'] += 1
                        behavior['direction_changes'] = 0
                        # Decay old restless detections
                        behavior['restless'] = min(behavior['restless'], self.behavior_history_length)

    def _is_person_idle(self, person_id: str) -> bool:
        """
        Check if a person has been idle (minimal movement) over the last frames.

        Args:
            person_id: Person identifier

        Returns:
            True if person is considered idle
        """
        if person_id not in self.person_motion_history:
            return False
        
        motion_history = self.person_motion_history[person_id]
        if len(motion_history) < self.motion_history_length:
            return False  # Not enough history to determine
        
        # Calculate total movement distance over the history
        total_movement = 0.0
        for i in range(1, len(motion_history)):
            prev_x, prev_y = motion_history[i-1]
            curr_x, curr_y = motion_history[i]
            distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
            total_movement += distance
        
        # If total movement is below threshold, consider idle
        return total_movement < self.idle_threshold

    def _determine_person_status(self, person_id: str) -> str:
        if person_id not in self.person_hand_active:
            self.person_hand_active[person_id] = False

        if person_id not in self.person_valid_zone:
            self.person_valid_zone[person_id] = False

        if person_id not in self.person_motion_streak:
            self.person_motion_streak[person_id] = 0

        if not hasattr(self, "person_state"):
            self.person_state = {}

        if person_id not in self.person_state:
            self.person_state[person_id] = "NORMAL"

        is_hand_active = self.person_hand_active.get(person_id, False)
        is_valid_zone = self.person_valid_zone.get(person_id, False)

        # avg motion from ROI history
        motion_history = self.motion_history.get(person_id, [])
        avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0.0

        # -------- VALID MOVEMENT CONDITION --------
        valid_movement = is_hand_active and is_valid_zone  # Threshold for meaningful motion

        if valid_movement:
            self.person_motion_streak[person_id] += 1
        else:
            # strong decay/reset so static hand won't stay suspicious
            self.person_motion_streak[person_id] = max(0, self.person_motion_streak[person_id] - 2)

        streak = self.person_motion_streak[person_id]

        # -------- FINAL STATE --------
        if streak >= 8:
            self.person_state[person_id] = "SUSPICIOUS"
            return "SUSPICIOUS"

        if streak >= 1:
            self.person_state[person_id] = "ACTIVE"
            return "ACTIVE"

        self.person_state[person_id] = "NORMAL"
        return "NORMAL"

    def _get_stable_person_status(self, person_id: str) -> str:
        """
        Apply stability logic to prevent flickering states and implement state memory.

        Rules:
        - Maintain status history (last 20 frames)
        - Only mark as SUSPICIOUS if suspicious for at least 10 frames in last 20
        - State memory prevents rapid downgrading:
          * SUSPICIOUS → require 10 normal frames to go back to NORMAL
          * ALERT → stay in ALERT until cooldown ends

        Args:
            person_id: Person identifier

        Returns:
            Stable status string: 'IDLE', 'NORMAL', 'ACTIVE', 'SUSPICIOUS', or 'ALERT'
        """
        # Get raw status from behavior analysis
        raw_status = self._determine_person_status(person_id)
        
        # Ensure status history exists for this person (avoid KeyError)
        status_history = self.person_status_history.setdefault(person_id, [])
        status_history.append(raw_status)
        
        # Keep only last N statuses
        if len(status_history) > self.status_history_length:
            status_history.pop(0)
        
        # If not enough history, return raw status
        if len(status_history) < 5:
            return raw_status
        
        # Manage continuous SUSPICIOUS counter
        if raw_status == 'SUSPICIOUS':
            self.person_suspicious_frame_counts[person_id] = self.person_suspicious_frame_counts.get(person_id, 0) + 1
        else:
            self.person_suspicious_frame_counts[person_id] = 0

        # Escalate to ALERT if SUSPICIOUS persists for 40 frames continuously
        if self.person_suspicious_frame_counts.get(person_id, 0) >= 40:
            status_history.append('ALERT')
            if len(status_history) > self.status_history_length:
                status_history.pop(0)
            return 'ALERT'

        # Count suspicious frames in recent history
        suspicious_count = status_history.count('SUSPICIOUS')
        
        # Determine current stable status using existing logic
        current_stable_status = None
        
        # Stability rules for SUSPICIOUS state
        if raw_status == 'SUSPICIOUS':
            # Need at least 10 suspicious frames in last 20 to mark as suspicious
            if suspicious_count >= self.suspicious_stability_threshold:
                current_stable_status = 'SUSPICIOUS'
            else:
                # Not stable enough, return the most common recent status
                recent_statuses = status_history[-10:]  # Last 10 frames
                if recent_statuses.count('ACTIVE') > recent_statuses.count('NORMAL'):
                    current_stable_status = 'ACTIVE'
                else:
                    current_stable_status = 'NORMAL'
        else:
            # Not currently suspicious, check if we should reset from previous suspicious state
            if suspicious_count >= self.suspicious_stability_threshold:
                # Still suspicious based on history, maintain SUSPICIOUS state briefly
                current_stable_status = 'SUSPICIOUS'
            else:
                # Normal behavior, return raw status
                current_stable_status = raw_status
        
        # Apply state memory rules to prevent rapid downgrading
        committed_status = self.person_committed_status.get(person_id)
        
        # Initialize committed status if this is the first time
        if committed_status is None:
            self.person_committed_status[person_id] = current_stable_status
            return current_stable_status
        
        # ALERT state: stay in ALERT until cooldown ends (handled elsewhere)
        if committed_status == 'ALERT':
            # Check if ALERT cooldown has expired (alert_active is False)
            if not self.alert_active:
                # ALERT cooldown ended, can downgrade
                self.person_committed_status[person_id] = current_stable_status
                return current_stable_status
            else:
                # Still in ALERT cooldown, stay ALERT
                return 'ALERT'
        
        # SUSPICIOUS state: require 10 normal frames to downgrade
        if committed_status == 'SUSPICIOUS':
            if current_stable_status in ['NORMAL', 'ACTIVE', 'IDLE']:
                # Count normal frames
                self.person_normal_frame_counts[person_id] = self.person_normal_frame_counts.get(person_id, 0) + 1
                if self.person_normal_frame_counts.get(person_id, 0) >= 10:
                    # Enough normal frames, downgrade
                    self.person_committed_status[person_id] = current_stable_status
                    self.person_normal_frame_counts[person_id] = 0  # Reset counter
                    return current_stable_status
                else:
                    # Not enough normal frames, stay SUSPICIOUS
                    return 'SUSPICIOUS'
            else:
                # Still suspicious or escalating, reset normal counter
                self.person_normal_frame_counts[person_id] = 0
                # Update committed status if escalating to ALERT
                if current_stable_status == 'ALERT':
                    self.person_committed_status[person_id] = 'ALERT'
                return current_stable_status
        
        # Other states: allow immediate upgrades, but prevent downgrades
        if current_stable_status in ['SUSPICIOUS', 'ALERT']:
            # Upgrading to higher alert level
            self.person_committed_status[person_id] = current_stable_status
            return current_stable_status
        else:
            # Staying in same or lower level
            self.person_committed_status[person_id] = current_stable_status
            return current_stable_status

    def _update_person_predictions(self, person_detections: List[PersonDetection], person_probabilities: List[float]) -> None:
        """
        Update per-person prediction buffers with stability mechanisms.

        Args:
            person_detections: List of detected persons
            person_probabilities: List of probabilities for each person
        """
        # Get current person IDs
        current_person_ids = set()

        for detection, probability in zip(person_detections, person_probabilities):
            person_id = self._get_person_id(detection.bbox)
            current_person_ids.add(person_id)

            # Initialize buffer if new person
            if person_id not in self.person_prediction_buffers:
                self.person_prediction_buffers[person_id] = []
                self.person_alert_counts[person_id] = 0

            # Add prediction to buffer (keep last 20)
            buffer = self.person_prediction_buffers[person_id]
            buffer.append(probability)
            if len(buffer) > 20:
                buffer.pop(0)

        # Reset buffers for persons no longer detected
        expired_persons = set(self.person_prediction_buffers.keys()) - current_person_ids
        for person_id in expired_persons:
            del self.person_prediction_buffers[person_id]
            if person_id in self.person_alert_counts:
                del self.person_alert_counts[person_id]
            if person_id in self.person_prediction_cache:
                del self.person_prediction_cache[person_id]

    def _check_sustained_alerts(self) -> Tuple[bool, bool, float]:
        has_suspicious = False
        has_alert = False
        max_suspicious_score = 0.0

        # Check all currently known persons from detections, not old movement states
        current_ids = set()

        for detection in self.last_person_detections:
            pid = self._get_person_id(detection.bbox)
            current_ids.add(pid)

            stable_status = self._get_stable_person_status(pid)

            if stable_status == "ALERT":
                has_alert = True
            elif stable_status == "SUSPICIOUS":
                has_suspicious = True

        # Clean stale counters for disappeared persons
        stale_ids = set(self.person_suspicious_frame_counts.keys()) - current_ids
        for pid in stale_ids:
            self.person_suspicious_frame_counts.pop(pid, None)
            self.person_active_start_times.pop(pid, None)
            self.person_last_hand_time.pop(pid, None)
            self.person_hand_active.pop(pid, None)

        return has_suspicious, has_alert, max_suspicious_score
        

    def calculate_motion(self, prev_frame, curr_frame):


        import cv2
        import numpy as np

        if prev_frame is None:
            return 0.0

        # Convert to gray
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Blur (noise reduce)
        prev_gray = cv2.GaussianBlur(prev_gray, (5,5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5,5), 0)

        # Frame difference
        diff = cv2.absdiff(prev_gray, curr_gray)

        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Motion score
        motion_score = np.sum(thresh) / 255

        return motion_score / (thresh.shape[0] * thresh.shape[1])
        

    def _process_frame_detection(self, frame: cv2.Mat) -> cv2.Mat:
        import time
    
        self.has_current_alert = False

        # mild contrast enhancement
        frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=8)
    
        original_h, original_w = frame.shape[:2]
    
        # detect on smaller frame
        processing_frame = cv2.resize(frame, (480, 360))

        current_detection_interval = 15 if self.alert_active else self.detection_interval

        if self.frame_counter % current_detection_interval == 0:
            # tracker returns boxes in processing_frame coordinates
            detected = self.tracker.detect_persons(processing_frame)

            # scale once from 480x360 -> actual display frame
            scale_x = frame.shape[1] / 480
            scale_y = frame.shape[0] / 360

            person_detections = []
            for detection in detected:
                x1, y1, x2, y2 = detection.bbox
                person_detections.append(
                    PersonDetection(
                        bbox=(
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(x2 * scale_x),
                            int(y2 * scale_y),
                        ),
                        confidence=detection.confidence,
                    )
                )
                        
            # Reset flags for all visible persons
            for detection in person_detections:
                pid = self._get_person_id(detection.bbox)
                self.person_hand_active[pid] = False
                self.person_valid_zone[pid] = False

            motion_points_copy = self.motion_points.copy()

            for (hx, hy) in motion_points_copy:
                matched = False

                for detection in person_detections:
                    x1, y1, x2, y2 = detection.bbox
                    pid = self._get_person_id(detection.bbox)

                    person_h = y2 - y1
                    person_w = x2 - x1

                    if person_h <= 0 or person_w <= 0:
                        continue
                    
                    # ---- chest block + pocket valid zone ----
                    chest_block_y2 = y1 + int(person_h * 0.60)
                    valid_zone_y1 = y1 + int(person_h * 0.63)
                    valid_zone_y2 = y1 + int(person_h * 0.95)

                    # horizontal zone widened slightly for pocket side
                    zone_x1 = x1 + int(person_w * 0.03)
                    zone_x2 = x2 - int(person_w * 0.03)

                    # reject chest / upper movement
                    if hy <= chest_block_y2:
                        continue
                    
                    # accept only waist/pocket band
                    if zone_x1 < hx < zone_x2 and valid_zone_y1 < hy < valid_zone_y2:
                        self.person_hand_active[pid] = True
                        self.person_valid_zone[pid] = True
                        matched = True
                        break
                    
                if not matched:
                    pass

            # build status + probabilities
            person_probabilities = []
            for detection in person_detections:
                pid = self._get_person_id(detection.bbox)
                status = self._get_stable_person_status(pid)
    
                if status == "SUSPICIOUS":
                    person_probabilities.append(0.8)
                else:
                    person_probabilities.append(0.0)
    
            with self.detection_lock:
                self.last_person_detections = person_detections
                self.last_person_probabilities = person_probabilities
    
        else:
            with self.detection_lock:
                person_detections = self.last_person_detections
                person_probabilities = self.last_person_probabilities
    
        # draw detections + trails
        if not self.alert_active:
            self.tracker.update_tracks(person_detections)
    
        self._draw_person_detections(frame, person_detections, person_probabilities)
        self.tracker.draw_tracking_trail(frame, self.tracker.track_history)
        self._draw_hand_trail(frame)
    
        # current state from currently visible persons only
        has_suspicious = False
        for detection in person_detections:
            pid = self._get_person_id(detection.bbox)
            status = self._get_stable_person_status(pid)
            if status in ("ACTIVE", "SUSPICIOUS", "ALERT"):
                has_suspicious = True
            if status in ("SUSPICIOUS", "ALERT"):
                self.has_current_alert = True
    
        max_probability = max(person_probabilities) if person_probabilities else 0.0
    
        current_time = time.time()
        current_state = "SUSPICIOUS" if has_suspicious else "NORMAL"
    
        if current_state == "SUSPICIOUS":
            self.suspicious_frames_count += 1
        else:
            self.suspicious_frames_count = 0
    
        cooldown_passed = current_time - self.last_alert_time > 5.0
    
        if self.has_current_alert and self.suspicious_frames_count >= 5 and cooldown_passed:
            logger.info(f"🚨 Suspicious activity confirmed ({self.suspicious_frames_count})")
            self._trigger_alert(frame, max_probability)
            self.last_alert_time = current_time
    
        self.previous_alert_state = current_state
        self.alert_state = current_state == "SUSPICIOUS"
    
        # overlay
        self._draw_overlay(frame, len(person_detections), max_probability)
        self._draw_alert_border(frame)
    
        return frame

    def run(self) -> None:
        """Run the detection loop with threading."""
        if not self.initialize():
            return

        self.running = True
        logger.info("Starting threaded detection system...")

        try:
            # Start camera capture thread
            self.capture_thread = threading.Thread(target=self._camera_capture_thread, daemon=True)
            self.capture_thread.start()
            
            # Start detection processing thread
            self.detection_thread = threading.Thread(target=self._detection_processing_thread, daemon=True)
            self.detection_thread.start()
            
            logger.info("Threads started, beginning display loop...")

            # Main display loop
            while self.running:
                # Get the latest processed frame for display
                with self.frame_lock:
                    if self.display_frame is not None:
                        display_frame = self.display_frame.copy()
                    else:
                        # Create a blank frame if no processed frame available yet
                        display_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                        # Add loading text
                        cv2.putText(display_frame, "Loading...", (config.FRAME_WIDTH//2 - 50, config.FRAME_HEIGHT//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display the frame
                cv2.imshow('AI Shoplifting Surveillance System', display_frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested")
                    break

                # Small delay to prevent excessive CPU usage in display loop
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Error in detection loop: {e}")
        finally:
            self.cleanup()

    def stop(self) -> None:
        """Stop the detection system."""
        self.running = False
        logger.info("Stopping detection system...")

    def cleanup(self) -> None:
        """Clean up resources and stop threads."""
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Clean up camera
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources cleaned up and threads stopped")


def main():
    """Main function."""
    # Use config module directly
    # Override config from command line if needed
    # For now, use defaults

    system = ShopliftingDetectionSystem()

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        system.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        system.run()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()