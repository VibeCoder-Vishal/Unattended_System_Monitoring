import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

class UnattendedMonitorDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initializes the detector with a YOLO model.

        Args:
            model_path (str, optional): Path to a custom YOLO model.
                                        If None, uses a default model.
            confidence_threshold (float, optional): Detection confidence threshold.
                                                   Defaults to 0.5.
        """
        self.confidence_threshold = confidence_threshold

        # Determine model to load
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model from: {model_path}")
            self.model = YOLO(model_path)
        else:
            if model_path: # Path provided but not found
                 print(f"Warning: Custom model not found at '{model_path}'.")
            # Try loading a default model potentially included with your project
            default_model_path = "models/yolo11x.pt" # Check if this model exists
            if os.path.exists(default_model_path):
                print(f"Loading default model from: {default_model_path}")
                self.model = YOLO(default_model_path)
            else:
                # Fallback to a standard YOLOv8 pretrained model (will download if needed)
                print("Loading standard pretrained YOLOv8n model.")
                self.model = YOLO("yolov8n.pt") # Use nano version as a fallback

        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device) # Ensure model is on the correct device
        print(f"Using device: {self.device}")

        # Class IDs based on COCO dataset (YOLOv8 default)
        self.monitor_class_ids = [62] # Class ID for 'tvmonitor'
        self.person_class_id = 0    # Class ID for 'person'

        # Tunable parameters (consider moving to a config file or args)
        self.brightness_threshold = 94
        self.std_dev_threshold = 67
        self.proximity_radius_multiplier = 0.5
        self.monitor_radius_multiplier = 0.1

    def detect_objects(self, frame):
        """
        Detects monitors and persons in the frame using the loaded YOLO model.

        Args:
            frame: The input image frame (NumPy array).

        Returns:
            tuple: (list_of_monitor_boxes, list_of_person_boxes)
                   Each box is (x1, y1, x2, y2, confidence).
        """
        monitors = []
        persons = []
        # Perform inference, disable verbose output
        results = self.model(frame, device=self.device, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf.item())
                if confidence >= self.confidence_threshold:
                    cls_id = int(box.cls.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Ensure coordinates are valid
                    if x1 < x2 and y1 < y2:
                        if cls_id in self.monitor_class_ids:
                            monitors.append((x1, y1, x2, y2, confidence))
                        elif cls_id == self.person_class_id:
                            persons.append((x1, y1, x2, y2, confidence))
        return monitors, persons

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

    def check_person_proximity(self, monitor_bbox, persons_list):
        """
        Checks if any person in the provided list is within a calculated
        proximity radius around the monitor.

        Args:
            monitor_bbox: Bounding box tuple of the monitor.
            persons_list: List of person bounding box tuples to check against.

        Returns:
            bool: True if a person is nearby, False otherwise.
        """
        mx1, my1, mx2, my2, _ = monitor_bbox
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
            px1, py1, px2, py2, _ = person
            # Standard Intersection over Union (IoU) style overlap check
            # Check if there is *no* overlap first (easier logic)
            no_overlap = (px2 < prox_x1 or  # Person is left of proximity area
                          px1 > prox_x2 or  # Person is right of proximity area
                          py2 < prox_y1 or  # Person is above proximity area
                          py1 > prox_y2)   # Person is below proximity area

            if not no_overlap:
                return True # Found an overlapping person

        return False # No person found in proximity