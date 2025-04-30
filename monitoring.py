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
            print(f"Loading custom monitor model from: {monitor_model_path}") #
            self.monitor_model = YOLO(monitor_model_path) #
        else:
            if monitor_model_path: # Path provided but not found #
                print(f"Warning: Custom monitor model not found at '{monitor_model_path}'.") #
            # Try loading a default model potentially included with your project
            # CHANGE: Use the specific model intended for monitors/general here
            default_monitor_model_path = "models/yolo11x.pt" # Example: your original default #
            if os.path.exists(default_monitor_model_path): #
                print(f"Loading default monitor model from: {default_monitor_model_path}") #
                self.monitor_model = YOLO(default_monitor_model_path) #
            else:
                 # Fallback to a standard YOLOv8 pretrained model (will download if needed)
                print("Loading standard pretrained YOLOv8n model as monitor model fallback.") #
                self.monitor_model = YOLO("yolov8n.pt") # Nano version as fallback #
        self.monitor_model.to(self.device) #
        print("Monitor model loaded.")

        # --- Load Person Model ---
        print("--- Loading Person Model ---") #
        if person_model_path and os.path.exists(person_model_path): #
            print(f"Loading custom person model from: {person_model_path}") #
            self.person_model = YOLO(person_model_path) #
        else:
            if person_model_path: # Path provided but not found #
                print(f"Warning: Custom person model not found at '{person_model_path}'.") #
            # Fallback to a standard YOLOv8 pretrained model (good for person detection)
            print("Loading standard pretrained YOLOv8n model as person model fallback.") #
            # You might want a different default person detector if needed
            self.person_model = YOLO("yolov8n.pt") #
        self.person_model.to(self.device) #
        print("Person model loaded.")

        # Class IDs based on COCO dataset (YOLOv8 default)
        # Ensure these match the classes your *specific* models are trained on
        self.monitor_class_ids = [0] # Class ID for 'tvmonitor', 'laptop' in COCO #
                                          # Adjust if your monitor model uses different IDs
        self.person_class_id = 1          # Class ID for 'person' in COCO #
                                          # Adjust if your person model uses a different ID

        # Tunable parameters (consider moving to a config file or args)
        self.brightness_threshold = 94 #
        self.std_dev_threshold = 67 #
        self.proximity_radius_multiplier = 0.5 #
        self.monitor_radius_multiplier = 0.1 #

        # Tracking state remains the same, tracking will be applied by the person model's call
        self.track_history = defaultdict(lambda: []) #
        self.frame_count = 0 #

    # MODIFY detect_objects to use tracking for BOTH models
    def detect_objects(self, frame):
        """
        Detects and tracks monitors using the monitor model and persons
        using the person model.

        Args:
            frame: The input image frame (NumPy array).
        Returns:
            tuple: (list_of_monitor_boxes, list_of_person_boxes)
                   Monitor box: (x1, y1, x2, y2, confidence, track_id). # MODIFIED
                   Person box: (x1, y1, x2, y2, confidence, track_id).
        """
        monitors = []
        persons = []

        # Increment frame counter for tracking consistency (used internally by tracker)
        self.frame_count += 1 #

        # --- 1. Detect and Track Monitors (using monitor_model) ---
        # CHANGE: Use track() instead of predict() for monitors
        # Use persist=True, filter by monitor classes
        monitor_results = self.monitor_model.track(
            frame,
            persist=True, # Crucial for tracking across frames
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
            classes=self.monitor_class_ids # Filter for monitor classes here #
        )

        # Extract monitor boxes with track IDs
        # The structure might vary slightly based on ultralytics version, adjust if needed.
        if len(monitor_results) > 0 and monitor_results[0].boxes.id is not None: #
            monitor_boxes = monitor_results[0].boxes.xyxy.cpu().numpy().astype(int) #
            monitor_track_ids = monitor_results[0].boxes.id.int().cpu().tolist() #
            monitor_confidences = monitor_results[0].boxes.conf.cpu().numpy() #
            # monitor_class_ids_detected = monitor_results[0].boxes.cls.cpu().numpy() # Already filtered

            for i, box_coords in enumerate(monitor_boxes):
                 x1, y1, x2, y2 = box_coords #
                 track_id = monitor_track_ids[i] #
                 confidence = monitor_confidences[i] #
                 # cls_id = monitor_class_ids_detected[i] # Should be in self.monitor_class_ids

                 if x1 < x2 and y1 < y2: # Ensure valid coordinates #
                     # CHANGE: Append track_id to the monitor tuple
                     monitors.append((x1, y1, x2, y2, confidence, track_id)) #
        else:
             # Fallback if tracking IDs are missing (e.g., no monitors detected)
             # You might add simple detection here if needed, but usually tracking provides IDs if objects exist
             pass # Or log a warning


        # --- 2. Detect and Track Persons (using person_model) ---
        # This part remains the same as before
        person_results = self.person_model.track( #
            frame,
            persist=True, #
            device=self.device, #
            verbose=False, #
            conf=self.confidence_threshold, #
            classes=[self.person_class_id] # Filter for person class #
        )

        if len(person_results) > 0 and person_results[0].boxes.id is not None: #
            tracked_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int) #
            track_ids = person_results[0].boxes.id.int().cpu().tolist() #
            confidences = person_results[0].boxes.conf.cpu().numpy() #

            for i, box_coords in enumerate(tracked_boxes): #
                 x1, y1, x2, y2 = box_coords #
                 track_id = track_ids[i] #
                 confidence = confidences[i] #

                 if x1 < x2 and y1 < y2: #
                     persons.append((x1, y1, x2, y2, confidence, track_id)) #
        else:
            # Fallback if tracking IDs are missing for persons
            pass # Or log a warning

        return monitors, persons

    # --- is_monitor_on remains unchanged ---
    # Note: it already uses *_ to unpack, so it should handle the extra track_id
    def is_monitor_on(self, frame, monitor_bbox):
        """
        Determines if a monitor is likely 'on' (not black/uniform)
        based on brightness/standard deviation in a central region.
        Args:
            frame: The input image frame.
            monitor_bbox: The bounding box tuple (x1, y1, x2, y2, conf, track_id). # Updated comment
        Returns:
            bool: True if the monitor seems 'on', False otherwise.
        """
        x1, y1, x2, y2, *_ = monitor_bbox # Use *_ to safely ignore extra elements like conf, track_id #
        # ... rest of the function is the same ...
        width = x2 - x1 #
        height = y2 - y1 #
        # Basic validation for bounding box dimensions
        if width <= 0 or height <= 0: #
            return False #
        # Calculate center and radius for analysis region
        center_x = (x1 + x2) // 2 #
        center_y = (y1 + y2) // 2 #
        radius = int(min(width, height) * self.monitor_radius_multiplier) #
        radius = max(1, radius) # Ensure radius is at least 1 pixel #
        # Create mask for the circular region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8) #
        cv2.circle(mask, (center_x, center_y), radius, 255, -1) #
        # Define safe boundaries for slicing the frame and mask
        y1_safe, y2_safe = max(0, y1), min(frame.shape[0], y2) #
        x1_safe, x2_safe = max(0, x1), min(frame.shape[1], x2) #
        # Extract regions only if safe coordinates are valid
        if y1_safe >= y2_safe or x1_safe >= x2_safe: #
             return False # Invalid region size #
        monitor_region = frame[y1_safe:y2_safe, x1_safe:x2_safe] #
        mask_region = mask[y1_safe:y2_safe, x1_safe:x2_safe] #
        # Check if extracted regions are valid
        if monitor_region.size == 0 or mask_region.size == 0: #
            return False #
        try:
            hsv = cv2.cvtColor(monitor_region, cv2.COLOR_BGR2HSV) #
            v_channel = hsv[:, :, 2] # Value channel (brightness) #
        except cv2.error: #
            # If color conversion fails (e.g., unexpected image format) treat as off
            return False #
        # Apply mask to get brightness values only within the circle
        # Ensure mask is applied correctly even if bbox is at edge
        masked_v = cv2.bitwise_and(v_channel, v_channel, mask=mask_region) #
        valid_pixels = masked_v[mask_region == 255] #
        if valid_pixels.size == 0: #
            return False # No pixels selected by mask #
        # Calculate metrics
        avg_brightness = np.mean(valid_pixels) #
        std_dev = np.std(valid_pixels) #
        # Check against thresholds
        is_on = avg_brightness > self.brightness_threshold or std_dev > self.std_dev_threshold #
        return is_on

    # --- check_person_proximity remains unchanged ---
    # Note: it already uses *_ to unpack, so it should handle the extra track_id
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
        # ... (This function already uses *_ for monitor_bbox, so no change needed here) ...
        mx1, my1, mx2, my2, *_ = monitor_bbox #
        # ... rest of the function is the same ...
        monitor_width = mx2 - mx1 #
        monitor_height = my2 - my1 #
        if monitor_width <= 0 or monitor_height <= 0: #
            return False # Cannot calculate proximity for invalid box #
        # Define proximity area around the monitor
        radius = max(monitor_width, monitor_height) * self.proximity_radius_multiplier #
        prox_x1 = max(0, int(mx1 - radius)) #
        prox_y1 = max(0, int(my1 - radius)) #
        prox_x2 = int(mx2 + radius) # No min check needed, frame boundary handled by overlap check #
        prox_y2 = int(my2 + radius) # No min check needed #
        # Check overlap with each person
        for person in persons_list: #
            px1, py1, px2, py2, *_ = person # Handle varying length person tuples (e.g. with/without track_id) #
            # Standard Intersection over Union (IoU) style overlap check
            # Check if there is *no* overlap first (easier logic)
            no_overlap = (px2 < prox_x1 or  # Person is left of proximity area #
                          px1 > prox_x2 or  # Person is right of proximity area #
                          py2 < prox_y1 or  # Person is above proximity area #
                          py1 > prox_y2)    # Person is below proximity area #
            if not no_overlap: #
                return True # Found an overlapping person #
        return False # No person found in proximity 