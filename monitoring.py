# monitoring.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from collections import defaultdict

class UnattendedMonitorDetector:
    # MODIFY __init__ to accept two model paths
    def __init__(self, monitor_model_path=None, person_model_path=None, confidence_threshold=0.5):
        """
        Initializes the detector with potentially separate YOLO models for monitors and persons.
        Args:
            monitor_model_path (str, optional): Path to a YOLO model primarily for monitor detection.
                                                Uses a default/pretrained if None.
            person_model_path (str, optional): Path to a YOLO model specifically for person detection.
                                               Uses a default/pretrained if None.
            confidence_threshold (float, optional): Detection confidence threshold. Defaults to 0.5.
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # --- Load Monitor Model ---
        print("--- Loading Monitor Model ---")
        if monitor_model_path and os.path.exists(monitor_model_path):
            print(f"Loading custom monitor model from: {monitor_model_path}")
            self.monitor_model = YOLO(monitor_model_path)
        else:
            if monitor_model_path: # Path provided but not found
                print(f"Warning: Custom monitor model not found at '{monitor_model_path}'.")
            # Try loading a default model potentially included with your project
            # CHANGE: Use the specific model intended for monitors/general here
            default_monitor_model_path = "models/yolo11x.pt" # Example: your original default
            if os.path.exists(default_monitor_model_path):
                print(f"Loading default monitor model from: {default_monitor_model_path}")
                self.monitor_model = YOLO(default_monitor_model_path)
            else:
                 # Fallback to a standard YOLOv8 pretrained model (will download if needed)
                print("Loading standard pretrained YOLOv8n model as monitor model fallback.")
                self.monitor_model = YOLO("yolov8n.pt") # Nano version as fallback
        self.monitor_model.to(self.device)
        print("Monitor model loaded.")

        # --- Load Person Model ---
        print("--- Loading Person Model ---")
        if person_model_path and os.path.exists(person_model_path):
            print(f"Loading custom person model from: {person_model_path}")
            self.person_model = YOLO(person_model_path)
        else:
            if person_model_path: # Path provided but not found
                print(f"Warning: Custom person model not found at '{person_model_path}'.")
            # Fallback to a standard YOLOv8 pretrained model (good for person detection)
            print("Loading standard pretrained YOLOv8n model as person model fallback.")
            # You might want a different default person detector if needed
            self.person_model = YOLO("yolov8n.pt")
        self.person_model.to(self.device)
        print("Person model loaded.")

        # Class IDs based on COCO dataset (YOLOv8 default)
        # Ensure these match the classes your *specific* models are trained on
        self.monitor_class_ids = [0] # Class ID for 'tvmonitor', 'laptop' in COCO
                                          # Adjust if your monitor model uses different IDs
        self.person_class_id = 1          # Class ID for 'person' in COCO
                                          # Adjust if your person model uses a different ID

        # Tunable parameters (consider moving to a config file or args)
        self.brightness_threshold = 94
        self.std_dev_threshold = 67
        self.proximity_radius_multiplier = 0.5
        self.monitor_radius_multiplier = 0.1

        # Tracking state remains the same, tracking will be applied by the person model's call
        self.track_history = defaultdict(lambda: [])
        self.frame_count = 0

    # MODIFY detect_objects to use both models
    def detect_objects(self, frame):
        """
        Detects monitors using the monitor model and persons (with tracking)
        using the person model.

        Args:
            frame: The input image frame (NumPy array).
        Returns:
            tuple: (list_of_monitor_boxes, list_of_person_boxes)
                   Monitor box: (x1, y1, x2, y2, confidence).
                   Person box: (x1, y1, x2, y2, confidence, track_id).
        """
        monitors = []
        persons = []

        # Increment frame counter for tracking consistency
        self.frame_count += 1

        # --- 1. Detect Monitors (using monitor_model) ---
        # We can use 'predict' if tracking isn't needed for monitors,
        # or 'track' if it doesn't hurt performance significantly. Let's use predict.
        monitor_results = self.monitor_model.predict(frame, device=self.device, verbose=False, conf=self.confidence_threshold)

        for result in monitor_results:
            boxes = result.boxes
            for box in boxes:
                # Confidence check is already done by the predict `conf` argument,
                # but double-checking doesn't hurt if needed.
                # confidence = float(box.conf.item()) # Redundant if conf set in predict
                cls_id = int(box.cls.item())
                if cls_id in self.monitor_class_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item()) # Get confidence anyway
                    if x1 < x2 and y1 < y2: # Ensure valid coordinates
                         monitors.append((x1, y1, x2, y2, confidence))


        # --- 2. Detect and Track Persons (using person_model) ---
        # Use 'track' here to get track IDs for persons. Persist=True is crucial.
        person_results = self.person_model.track(frame, persist=True, device=self.device, verbose=False, conf=self.confidence_threshold, classes=[self.person_class_id]) # Filter classes here for efficiency

        # Check if tracking IDs are available in the results
        # The exact structure might vary slightly based on ultralytics version, adjust if needed.
        if len(person_results) > 0 and person_results[0].boxes.id is not None:
            tracked_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = person_results[0].boxes.id.int().cpu().tolist()
            confidences = person_results[0].boxes.conf.cpu().numpy()
            # class_ids = person_results[0].boxes.cls.cpu().numpy() # Already filtered by classes=[0]

            for i, box_coords in enumerate(tracked_boxes):
                 x1, y1, x2, y2 = box_coords
                 track_id = track_ids[i]
                 confidence = confidences[i]
                 # cls_id = class_ids[i] # Should always be self.person_class_id

                 if x1 < x2 and y1 < y2: # Ensure valid coordinates
                     persons.append((x1, y1, x2, y2, confidence, track_id))
                     # Optional: Update track history if needed elsewhere
                     # center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                     # self.track_history[track_id].append((center_x, center_y))
                     # if len(self.track_history[track_id]) > 30: # Limit history length
                     #      self.track_history[track_id].pop(0)
        else:
            # Fallback if tracking IDs are missing (e.g., no persons detected or tracker issue)
            # Run simple detection if tracking fails
            plain_person_results = self.person_model.predict(frame, device=self.device, verbose=False, conf=self.confidence_threshold, classes=[self.person_class_id])
            for result in plain_person_results:
                boxes = result.boxes
                for box in boxes:
                     cls_id = int(box.cls.item()) # Should be person_class_id
                     if cls_id == self.person_class_id:
                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                         confidence = float(box.conf.item())
                         if x1 < x2 and y1 < y2:
                             persons.append((x1, y1, x2, y2, confidence)) # No track_id here


        return monitors, persons

    # --- is_monitor_on remains unchanged ---
    def is_monitor_on(self, frame, monitor_bbox):
        """
        Determines if a monitor is likely 'on' (not black/uniform)
        based on brightness/standard deviation in a central region.
        Args:
            frame: The input image frame.
            monitor_bbox: The bounding box tuple (x1, y1, x2, y2, conf).
        Returns:
            bool: True if the monitor seems 'on', False otherwise.
        """
        x1, y1, x2, y2, _ = monitor_bbox
        width = x2 - x1
        height = y2 - y1
        # Basic validation for bounding box dimensions
        if width <= 0 or height <= 0:
            return False
        # Calculate center and radius for analysis region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = int(min(width, height) * self.monitor_radius_multiplier)
        radius = max(1, radius) # Ensure radius is at least 1 pixel
        # Create mask for the circular region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        # Define safe boundaries for slicing the frame and mask
        y1_safe, y2_safe = max(0, y1), min(frame.shape[0], y2)
        x1_safe, x2_safe = max(0, x1), min(frame.shape[1], x2)
        # Extract regions only if safe coordinates are valid
        if y1_safe >= y2_safe or x1_safe >= x2_safe:
             return False # Invalid region size
        monitor_region = frame[y1_safe:y2_safe, x1_safe:x2_safe]
        mask_region = mask[y1_safe:y2_safe, x1_safe:x2_safe]
        # Check if extracted regions are valid
        if monitor_region.size == 0 or mask_region.size == 0:
            return False
        try:
            hsv = cv2.cvtColor(monitor_region, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2] # Value channel (brightness)
        except cv2.error:
            # If color conversion fails (e.g., unexpected image format) treat as off
            return False
        # Apply mask to get brightness values only within the circle
        # Ensure mask is applied correctly even if bbox is at edge
        masked_v = cv2.bitwise_and(v_channel, v_channel, mask=mask_region)
        valid_pixels = masked_v[mask_region == 255]
        if valid_pixels.size == 0:
            return False # No pixels selected by mask
        # Calculate metrics
        avg_brightness = np.mean(valid_pixels)
        std_dev = np.std(valid_pixels)
        # Check against thresholds
        is_on = avg_brightness > self.brightness_threshold or std_dev > self.std_dev_threshold
        return is_on

    # --- check_person_proximity remains unchanged ---
    def check_person_proximity(self, monitor_bbox, persons_list):
        """
        Checks if any person in the provided list is within a calculated
        proximity radius around the monitor.
        Args:
            monitor_bbox: Bounding box tuple of the monitor.
            persons_list: List of person bounding box tuples to check against.
                           (May include confidence and track_id).
        Returns:
            bool: True if a person is nearby, False otherwise.
        """
        mx1, my1, mx2, my2, *_ = monitor_bbox # Use *_ to handle potential extra elements like confidence
        monitor_width = mx2 - mx1
        monitor_height = my2 - my1
        if monitor_width <= 0 or monitor_height <= 0:
            return False # Cannot calculate proximity for invalid box
        # Define proximity area around the monitor
        radius = max(monitor_width, monitor_height) * self.proximity_radius_multiplier
        prox_x1 = max(0, int(mx1 - radius))
        prox_y1 = max(0, int(my1 - radius))
        prox_x2 = int(mx2 + radius) # No min check needed, frame boundary handled by overlap check
        prox_y2 = int(my2 + radius) # No min check needed
        # Check overlap with each person
        for person in persons_list:
            px1, py1, px2, py2, *_ = person # Handle varying length person tuples (e.g. with/without track_id)
            # Standard Intersection over Union (IoU) style overlap check
            # Check if there is *no* overlap first (easier logic)
            no_overlap = (px2 < prox_x1 or  # Person is left of proximity area
                          px1 > prox_x2 or  # Person is right of proximity area
                          py2 < prox_y1 or  # Person is above proximity area
                          py1 > prox_y2)    # Person is below proximity area
            if not no_overlap:
                return True # Found an overlapping person
        return False # No person found in proximity