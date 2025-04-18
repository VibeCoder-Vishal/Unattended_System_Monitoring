import cv2
import numpy as np
from ultralytics import YOLO
import torch

class UnattendedMonitorDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize the UnattendedMonitorDetector class
        
        Args:
            model_path: Path to a custom YOLO model (if None, uses pretrained YOLOv8)
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("models/yolo11x.pt")  # n for nano - smallest and fastest
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Classes of interest - In YOLOv8, "tv" is class 62
        self.monitor_class_ids = [62]  # TV/monitor class ID
        self.person_class_id = 0       # Person class ID
        
        # Monitor state thresholds
        self.brightness_threshold = 94  # Lower threshold for "non-black" monitors
        self.std_dev_threshold = 67     # Standard deviation threshold for "non-black" monitors
        
        # Association parameters
        self.proximity_radius_multiplier = 0.2  # Multiplier for proximity radius around monitors
        
        # NEW: Radius multiplier for circular region in monitor state analysis
        self.monitor_radius_multiplier = 0.1  # Multiplier for circular region radius
        
    def detect_objects(self, frame):
        """
        Detect monitors and persons in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            monitors: List of monitor bounding boxes [x1, y1, x2, y2, confidence]
            persons: List of person bounding boxes [x1, y1, x2, y2, confidence]
        """
        # Run YOLOv8 inference
        results = self.model(frame)
        
        # Parse results
        monitors = []
        persons = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get class ID, confidence, and box coordinates
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Add to appropriate list based on class
                if cls_id in self.monitor_class_ids:
                    monitors.append((x1, y1, x2, y2, confidence))
                elif cls_id == self.person_class_id:
                    persons.append((x1, y1, x2, y2, confidence))
        
        return monitors, persons
    
    def is_monitor_on(self, frame, monitor_bbox):
        """
        Determine if a monitor is turned on based on brightness and variation
        within a circular region at the midpoint of the bounding box.
        
        Args:
            frame: Input image frame
            monitor_bbox: Monitor bounding box [x1, y1, x2, y2, conf]
            
        Returns:
            Boolean indicating if monitor is on (non-black)
        """
        x1, y1, x2, y2, _ = monitor_bbox
        
        # Calculate monitor dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate midpoint
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        
        # Define radius as a fraction of the smaller dimension
        radius = int(min(width, height) * self.monitor_radius_multiplier)
        
        # Create a mask for the circular region
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (mx, my), radius, 255, -1)
        
        # Extract the monitor region (bounding box for safety)
        monitor_region = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        mask_region = mask[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        
        if monitor_region.size == 0 or mask_region.size == 0:
            return False
        
        # Convert monitor region to HSV
        hsv = cv2.cvtColor(monitor_region, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Apply mask to Value channel (relative to bounding box)
        masked_v = cv2.bitwise_and(v_channel, v_channel, mask=mask_region)
        
        # Get pixels within the circular region
        valid_pixels = masked_v[mask_region == 255]
        
        if valid_pixels.size == 0:
            return False
        
        # Calculate brightness metrics
        avg_brightness = np.mean(valid_pixels)
        std_dev = np.std(valid_pixels)
        
        # Determine if monitor is on (non-black)
        is_on = avg_brightness > self.brightness_threshold or std_dev > self.std_dev_threshold
        
        return is_on
    
    def check_person_proximity(self, monitor_bbox, persons):
        """
        Check if any person is in proximity to the monitor
        
        Args:
            monitor_bbox: Monitor bounding box [x1, y1, x2, y2, conf]
            persons: List of person bounding boxes [x1, y1, x2, y2, conf]
            
        Returns:
            Boolean indicating if any person is close to the monitor
        """
        mx1, my1, mx2, my2, _ = monitor_bbox
        monitor_width = mx2 - mx1
        monitor_height = my2 - my1
        
        # Calculate proximity radius
        radius = max(monitor_width, monitor_height) * self.proximity_radius_multiplier
        
        # Create expanded monitor box
        expanded_mx1 = max(0, int(mx1 - radius))
        expanded_my1 = max(0, int(my1 - radius))
        expanded_mx2 = int(mx2 + radius)
        expanded_my2 = int(my2 + radius)
        
        # Check if any person box overlaps with expanded monitor box
        for person in persons:
            px1, py1, px2, py2, _ = person
            
            # Check for overlap
            if not (px2 < expanded_mx1 or px1 > expanded_mx2 or py2 < expanded_my1 or py1 > expanded_my2):
                return True
                
        return False
    
    def detect_unattended_monitors(self, frame):
        """
        Detect unattended monitors in a frame
        
        Args:
            frame: Input image frame
            
        Returns:
            unattended_monitors: List of unattended monitor bounding boxes
            total_monitors_on: Total number of monitors that are on
            frame_with_annotations: Annotated frame with detections
        """
        # Make a copy for annotations
        annotated_frame = frame.copy()
        
        # Detect objects
        monitors, persons = self.detect_objects(frame)
        
        # Track statistics
        unattended_monitors = []
        total_monitors_on = 0
        
        # Process each monitor
        for monitor in monitors:
            x1, y1, x2, y2, conf = monitor
            
            # Check if monitor is on (non-black)
            is_on = self.is_monitor_on(frame, monitor)
            
            if is_on:
                total_monitors_on += 1
                
                # Check if person is nearby
                is_attended = self.check_person_proximity(monitor, persons)
                
                if not is_attended:
                    unattended_monitors.append(monitor)
                    # Draw red box for unattended monitor
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "Unattended (on)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Draw green box for attended monitor
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Attended", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw red box for off/black monitor
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Unattended (Off)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                unattended_monitors.append(monitor)
            
        # Draw person bounding boxes
        for person in persons:
            x1, y1, x2, y2, _ = person
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange for persons
            
        # Display statistics
        cv2.putText(annotated_frame, f"Unattended Monitors: {len(unattended_monitors)}/{len(monitors)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                   
        return unattended_monitors, total_monitors_on, annotated_frame