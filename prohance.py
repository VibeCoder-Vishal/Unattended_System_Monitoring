# prohance.py

import cv2
import numpy as np
import time


# --- Configuration Import ---
# Import thresholds from config.py. Define defaults if import fails.
try:
    from config import (
        SHORT_INTERACTION_THRESHOLD_SEC,
        SUSTAINED_PRESENCE_THRESHOLD_SEC,
        MANIPULATION_COOLDOWN_SEC
        # Add other config imports if needed, e.g., DEEPSORT_WEIGHTS
    )
    print("Prohance thresholds loaded from config.py")
except ImportError:
    print("Warning: Could not import Prohance thresholds from config.py. Using default values.")
    SHORT_INTERACTION_THRESHOLD_SEC = 10  # Default: 10 seconds
    SUSTAINED_PRESENCE_THRESHOLD_SEC = 60 # Default: 60 seconds
    MANIPULATION_COOLDOWN_SEC = 30      # Default: 30 seconds

# --- Mock DeepSORT Tracker ---
# Replace this with your actual DeepSORT (or other tracker) integration.
# This mock version assigns new IDs every frame for demonstration.
class MockDeepSORT:
    def __init__(self, model_weights_path=None, max_age=30, nn_budget=None, nms_max_overlap=1.0):
        self.next_track_id = 0
        # In a real tracker: Load model weights, initialize parameters like max_age etc.
        print(f"MockDeepSORT Initialized (Ignoring weights/params)")

    def update(self, frame, detections_xywh, confidences):
        """
        Mocks the update process. Assigns new IDs sequentially.
        Args:
            frame: The current frame.
            detections_xywh (list): Detections in [center_x, center_y, w, h] format.
            confidences (list): Detection confidences.
        Returns:
            list: Tracked objects in [[x1, y1, x2, y2, track_id], ...].
        """
        tracked_objects = []
        if not detections_xywh:
            return tracked_objects

        for i, (cx, cy, w, h) in enumerate(detections_xywh):
            # Convert xywh to x1y1x2y2
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Assign a new ID sequentially in this mock version
            self.next_track_id += 1
            track_id = self.next_track_id
            tracked_objects.append([x1, y1, x2, y2, track_id])

        return tracked_objects
# --- End Mock DeepSORT ---


class ProhanceMonitor:
    """
    Manages ROI states, tracks occupants, and detects Prohance-specific events
    like manipulation and sustained foreign occupancy.
    """

    # Define state labels
    STATE_INITIALIZING = "INITIALIZING"
    STATE_EMPTY_OFF = "EMPTY_OFF"
    STATE_EMPTY_ON = "EMPTY_ON"
    STATE_ATTENDED_PRIMARY = "ATTENDED_PRIMARY"
    STATE_ATTENDED_FOREIGN_BRIEF = "ATTENDED_FOREIGN_BRIEF"
    STATE_ATTENDED_FOREIGN_SUSTAINED = "ATTENDED_FOREIGN_SUSTAINED"
    STATE_UNKNOWN = "UNKNOWN"

    # Define colors for annotations (BGR format)
    COLOR_MAP = {
        STATE_INITIALIZING: (128, 128, 128), # Grey
        STATE_EMPTY_OFF: (100, 100, 100),    # Dark Grey
        STATE_EMPTY_ON: (0, 165, 255),       # Orange (Vulnerable)
        STATE_ATTENDED_PRIMARY: (0, 255, 0), # Green
        STATE_ATTENDED_FOREIGN_BRIEF: (0, 255, 255), # Yellow
        STATE_ATTENDED_FOREIGN_SUSTAINED: (0, 0, 255), # Red (Alert)
        STATE_UNKNOWN: (255, 0, 255),        # Magenta
    }
    PERSON_BOX_COLOR = (255, 0, 0) # Blue for tracked persons
    MANIPULATION_EVENT_COLOR = (0, 0, 255) # Red for manipulation alert text/highlight

    def __init__(self, num_rois, rois_np, thresholds_config, primary_users=None):
        """
        Initializes the ProhanceMonitor.

        Args:
            num_rois (int): Number of ROIs.
            rois_np (list): List of ROI polygons as NumPy arrays.
            thresholds_config (module or dict): Config object/dict containing thresholds like:
                                               SHORT_INTERACTION_THRESHOLD_SEC,
                                               SUSTAINED_PRESENCE_THRESHOLD_SEC,
                                               MANIPULATION_COOLDOWN_SEC.
            primary_users (dict, optional): Dictionary mapping ROI index to the
                                            track_id of the primary user for that ROI.
                                            e.g., {0: 1, 1: 5}. Defaults to None.
        """
        self.num_rois = num_rois
        self.rois_np = rois_np
        self.thresholds = {
            'short_interaction': thresholds_config.SHORT_INTERACTION_THRESHOLD_SEC,
            'sustained_presence': thresholds_config.SUSTAINED_PRESENCE_THRESHOLD_SEC,
            'manipulation_cooldown': thresholds_config.MANIPULATION_COOLDOWN_SEC,
        }
        self.primary_users = primary_users if primary_users else {}

        # --- Initialize Tracker ---
        # Replace MockDeepSORT with your actual tracker class
        # tracker_weights = getattr(thresholds_config, 'DEEPSORT_WEIGHTS', None) # Example
        self.tracker = MockDeepSORT() # Pass necessary args like weights path
        print("ProhanceMonitor initialized.")

        # --- Initialize ROI States ---
        self.roi_states = {}
        for i in range(self.num_rois):
            self.roi_states[i] = {
                'monitor_status': 'UNKNOWN', # ON, OFF, UNKNOWN
                'primary_occupant_id': self.primary_users.get(i, None),
                # {track_id: {'entry_time': timestamp}}
                'current_occupants': {},
                # {track_id: {'entry_time': timestamp, 'exit_time': timestamp}} - Keep recent exits
                'last_seen_occupants': {},
                'state_label': self.STATE_INITIALIZING,
                'last_state_change_time': time.time(),
                'manipulation_cooldown_end': 0, # Timestamp until next manipulation flag allowed
                'sustained_foreign_flagged': False, # Avoid repeated sustained events
                'last_manipulation_event_time': 0 # Track last manip event for annotation highlight
            }

    def _get_centroid(self, bbox):
        """Calculates the center point (x, y) of a bounding box."""
        x1, y1, x2, y2 = bbox[:4]
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _is_point_in_roi(self, point, roi_index):
        """Checks if a point is inside the specified ROI polygon."""
        if roi_index < 0 or roi_index >= len(self.rois_np):
            return False
        roi_polygon = self.rois_np[roi_index]
        # pointPolygonTest returns +ve if inside, 0 if on boundary, -ve if outside
        return cv2.pointPolygonTest(roi_polygon, point, False) >= 0

    def update_tracker(self, frame, person_detections):
        """
        Updates the object tracker with new person detections.

        Args:
            frame: The current video frame.
            person_detections (list): List of detected person bounding boxes from
                                     UnattendedMonitorDetector.detect_objects.
                                     Format: [(x1, y1, x2, y2, confidence), ...]

        Returns:
            list: List of tracked persons. Format: [[x1, y1, x2, y2, track_id], ...]
        """
        if not person_detections:
            # Pass empty lists to tracker if no detections
            return self.tracker.update(frame, [], [])

        # Convert detections format for the tracker (example assumes DeepSORT needs xywh, conf)
        # Adjust this based on your specific tracker's requirements!
        detections_xywh = []
        confidences = []
        for (x1, y1, x2, y2, conf) in person_detections:
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            detections_xywh.append([cx, cy, w, h])
            confidences.append(conf)

        # Update the tracker
        tracked_objects = self.tracker.update(frame, detections_xywh, confidences)

        return tracked_objects # Expected format: [[x1, y1, x2, y2, track_id], ...]

    def update_roi_states(self, frame_timestamp, tracked_persons, roi_monitor_statuses):
        """
        Updates the state of each ROI based on tracked occupants and monitor status,
        and detects Prohance-specific events.

        Args:
            frame_timestamp (float): Current timestamp (e.g., time.time()).
            tracked_persons (list): List of tracked persons from update_tracker.
                                    Format: [[x1, y1, x2, y2, track_id], ...]
            roi_monitor_statuses (dict): Dictionary mapping ROI index to its
                                         monitor status ('ON', 'OFF', 'UNKNOWN').
                                         e.g., {0: 'ON', 1: 'OFF'}

        Returns:
            list: A list of detected event dictionaries. Each event is like:
                  {'timestamp': float, 'roi': int, 'type': str,
                   'person_id': int or None, 'details': str}
        """
        detected_events = []
        persons_in_roi = {i: [] for i in range(self.num_rois)} # Store [track_id, bbox]

        # 1. Assign tracked persons to ROIs
        for person_track in tracked_persons:
            x1, y1, x2, y2, track_id = person_track
            centroid = self._get_centroid(person_track)
            for i in range(self.num_rois):
                if self._is_point_in_roi(centroid, i):
                    persons_in_roi[i].append({'id': track_id, 'bbox': person_track[:4]})
                    # Optional: break if a person can only be in one ROI

        # 2. Update state for each ROI
        for i in range(self.num_rois):
            state = self.roi_states[i]
            previous_state_label = state['state_label']
            current_monitor_status = roi_monitor_statuses.get(i, 'UNKNOWN')

            # Update monitor status in state
            state['monitor_status'] = current_monitor_status

            # Get current occupants' IDs for this ROI
            current_occupant_ids = {p['id'] for p in persons_in_roi[i]}
            previous_occupant_ids = set(state['current_occupants'].keys())

            # Identify newly entered and exited occupants
            entered_ids = current_occupant_ids - previous_occupant_ids
            exited_ids = previous_occupant_ids - current_occupant_ids

            # Update current_occupants and last_seen_occupants
            for track_id in exited_ids:
                occupant_data = state['current_occupants'].pop(track_id)
                occupant_data['exit_time'] = frame_timestamp
                state['last_seen_occupants'][track_id] = occupant_data
                # Simple cleanup: Remove very old entries from last_seen
                # (adjust threshold as needed)
                old_threshold = frame_timestamp - (self.thresholds['manipulation_cooldown'] * 5)
                state['last_seen_occupants'] = {
                    tid: data for tid, data in state['last_seen_occupants'].items()
                    if data.get('exit_time', 0) > old_threshold
                }


            for track_id in entered_ids:
                state['current_occupants'][track_id] = {'entry_time': frame_timestamp}
                # If this person was recently seen, remove from last_seen
                if track_id in state['last_seen_occupants']:
                    del state['last_seen_occupants'][track_id]

            # --- State Machine Logic ---
            current_state_label = self.STATE_UNKNOWN # Default
            primary_present = state['primary_occupant_id'] in current_occupant_ids
            foreign_occupants = {tid: data for tid, data in state['current_occupants'].items() if tid != state['primary_occupant_id']}
            has_foreign = bool(foreign_occupants)
            has_occupants = bool(state['current_occupants'])

            if not has_occupants:
                if current_monitor_status == 'ON':
                    current_state_label = self.STATE_EMPTY_ON
                elif current_monitor_status == 'OFF':
                    current_state_label = self.STATE_EMPTY_OFF
                else:
                     current_state_label = self.STATE_UNKNOWN # Monitor Unknown, no people
            elif primary_present:
                # If primary is present, consider it attended regardless of others or monitor status (for label)
                current_state_label = self.STATE_ATTENDED_PRIMARY
            elif has_foreign: # Only foreign occupants present
                 if current_monitor_status == 'ON':
                    # Check duration of the longest present foreign occupant
                    max_foreign_duration = 0
                    longest_foreign_occupant_id = None
                    if foreign_occupants:
                       max_foreign_duration = max(frame_timestamp - data['entry_time'] for data in foreign_occupants.values())


                    if max_foreign_duration >= self.thresholds['sustained_presence']:
                        current_state_label = self.STATE_ATTENDED_FOREIGN_SUSTAINED
                    else:
                        current_state_label = self.STATE_ATTENDED_FOREIGN_BRIEF
                 else: # Monitor OFF, but foreign people present - maybe just walking by?
                     # Treat as EMPTY_OFF for simplicity? Or a new state? Let's use UNKNOWN for now.
                     current_state_label = self.STATE_UNKNOWN # Or maybe 'FOREIGN_PRESENT_OFF'


            # --- Event Detection ---
            state_changed = (current_state_label != previous_state_label)
            if state_changed:
                state['last_state_change_time'] = frame_timestamp
                # Reset sustained flag if no longer in foreign sustained state
                if current_state_label != self.STATE_ATTENDED_FOREIGN_SUSTAINED:
                    state['sustained_foreign_flagged'] = False

                # Log primary user departure/return based on state change
                if previous_state_label == self.STATE_ATTENDED_PRIMARY and not primary_present:
                     detected_events.append({
                        'timestamp': frame_timestamp, 'roi': i,
                        'type': 'PRIMARY_DEPARTURE', 'person_id': state['primary_occupant_id'],
                        'details': f"Primary user {state['primary_occupant_id']} left ROI {i}. New state: {current_state_label}"
                    })
                elif previous_state_label != self.STATE_ATTENDED_PRIMARY and primary_present:
                    detected_events.append({
                        'timestamp': frame_timestamp, 'roi': i,
                        'type': 'PRIMARY_RETURN', 'person_id': state['primary_occupant_id'],
                        'details': f"Primary user {state['primary_occupant_id']} returned to ROI {i}. Previous state: {previous_state_label}"
                    })

            # Detect Sustained Foreign Occupancy Event (only log once)
            if current_state_label == self.STATE_ATTENDED_FOREIGN_SUSTAINED and not state['sustained_foreign_flagged']:
                 longest_foreign_occupant_id = None
                 if foreign_occupants:
                    # Find ID of one of the occupants triggering the sustained state
                    longest_foreign_occupant_id = max(foreign_occupants, key=lambda k: frame_timestamp - foreign_occupants[k]['entry_time'])

                 detected_events.append({
                    'timestamp': frame_timestamp, 'roi': i,
                    'type': 'SUSTAINED_FOREIGN_OCCUPANCY', 'person_id': longest_foreign_occupant_id, # Log one associated ID
                    'details': f"ROI {i} sustained foreign occupancy detected. User(s): {list(foreign_occupants.keys())}"
                })
                 state['sustained_foreign_flagged'] = True # Set flag to prevent re-logging immediately

            # Detect Manipulation Event (based on recent exits)
            for track_id, exit_data in list(state['last_seen_occupants'].items()): # Iterate copy for safe removal
                if track_id == state['primary_occupant_id']:
                    continue # Ignore primary user exits for manipulation check

                exit_time = exit_data.get('exit_time', 0)
                entry_time = exit_data.get('entry_time', 0)

                # Check if exit just happened and duration was short
                if exit_time == frame_timestamp and entry_time > 0: # Ensure valid entry time
                    duration = exit_time - entry_time
                    if duration < self.thresholds['short_interaction']:
                        # Check if cooldown allows logging
                        if frame_timestamp >= state['manipulation_cooldown_end']:
                             # Check if the state *before* or *during* the interaction was vulnerable (EMPTY_ON)
                             # This check is approximate; could refine by storing state history if needed
                             if previous_state_label == self.STATE_EMPTY_ON or previous_state_label == self.STATE_ATTENDED_FOREIGN_BRIEF:
                                detected_events.append({
                                    'timestamp': frame_timestamp, 'roi': i,
                                    'type': 'MANIPULATION_SUSPECTED', 'person_id': track_id,
                                    'details': f"ROI {i}: Brief interaction ({duration:.1f}s) by non-primary user {track_id} at vulnerable station."
                                })
                                state['manipulation_cooldown_end'] = frame_timestamp + self.thresholds['manipulation_cooldown']
                                state['last_manipulation_event_time'] = frame_timestamp
                                # Remove from last_seen once processed for manipulation to avoid re-triggering
                                del state['last_seen_occupants'][track_id]
                        # else: manipulation detected but in cooldown period

            # Update the state label
            state['state_label'] = current_state_label

        return detected_events


    def get_roi_annotations(self):
        """
        Generates annotation data (labels, colors, occupants) for each ROI.

        Returns:
            dict: Dictionary mapping ROI index to annotation details.
                  e.g., {0: {'label': 'ATTENDED_PRIMARY', 'color': (0,255,0), 'occupants': [1, 3]}, ...}
        """
        annotations = {}
        current_time = time.time()
        for i, state in self.roi_states.items():
            label = state['state_label']
            color = self.COLOR_MAP.get(label, self.COLOR_MAP[self.STATE_UNKNOWN])
            occupant_ids = list(state['current_occupants'].keys())
            primary_id = state['primary_occupant_id']
            
            # Add primary user ID to label if present and state allows
            display_label = f"ROI {i}: {label}"
            if label == self.STATE_ATTENDED_PRIMARY and primary_id is not None:
                 display_label += f" (P:{primary_id})"
                 # Optionally list other occupants:
                 # other_occupants = [oid for oid in occupant_ids if oid != primary_id]
                 # if other_occupants: display_label += f" O:{other_occupants}"
            elif label in [self.STATE_ATTENDED_FOREIGN_BRIEF, self.STATE_ATTENDED_FOREIGN_SUSTAINED]:
                 display_label += f" ({occupant_ids})"


            # Highlight briefly if a manipulation event just occurred
            if current_time < state['last_manipulation_event_time'] + 2.0: # Highlight for 2 seconds
                 # Could change color or add text like "[!]"
                 display_label = f"[!] {display_label}"
                 # Optionally use a different color temporarily
                 # color = self.MANIPULATION_EVENT_COLOR


            annotations[i] = {
                'label': display_label,
                'color': color,
                'occupants': occupant_ids,
                'primary_id': primary_id
            }
        return annotations

    def draw_annotations(self, frame, tracked_persons):
         """
         Draws ROI state labels and tracked person boxes onto the frame.
         (Helper function, could also be implemented in app.py using get_roi_annotations data)

         Args:
             frame: The frame to draw on (modified in place).
             tracked_persons (list): List of tracked persons [[x1, y1, x2, y2, track_id], ...].
         """
         roi_annotations = self.get_roi_annotations()

         # Draw ROI State Labels
         for i, roi_poly in enumerate(self.rois_np):
             if i in roi_annotations:
                 annotation = roi_annotations[i]
                 label = annotation['label']
                 color = annotation['color']

                 # Calculate position for the label (e.g., top-left corner or centroid)
                 try:
                      M = cv2.moments(roi_poly)
                      if M["m00"] != 0:
                          cx = int(M["m10"] / M["m00"])
                          cy = int(M["m01"] / M["m00"])
                      else: # Fallback for invalid moments
                          cx, cy = roi_poly[0][0] # Use first point
                      label_pos = (cx - 50, cy - 10) # Adjust offset as needed
                 except Exception:
                      label_pos = (roi_poly[0][0][0], roi_poly[0][0][1] - 10) # Fallback position


                 # Draw text with background
                 (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                 cv2.rectangle(frame, (label_pos[0], label_pos[1] - h - 5), (label_pos[0] + w + 5, label_pos[1] + 5), (0,0,0), -1)
                 cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

         # Draw Tracked Person Boxes and IDs
         persons_drawn_in_roi = {i: [] for i in range(self.num_rois)}
         for x1, y1, x2, y2, track_id in tracked_persons:
             centroid = self._get_centroid([x1,y1,x2,y2])
             in_any_monitored_roi = False
             for roi_idx in range(self.num_rois):
                  if self._is_point_in_roi(centroid, roi_idx):
                       in_any_monitored_roi = True
                       persons_drawn_in_roi[roi_idx].append(track_id)
                       # Draw only if inside a defined ROI
                       cv2.rectangle(frame, (x1, y1), (x2, y2), self.PERSON_BOX_COLOR, 2)
                       cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.PERSON_BOX_COLOR, 2)
                       break # Draw once even if in overlapping ROIs