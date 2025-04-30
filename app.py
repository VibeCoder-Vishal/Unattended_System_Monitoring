# app.py
import cv2
import numpy as np
import argparse
import os
import time
import sys
from collections import defaultdict # Import defaultdict

# Import components from other modules
# Ensure monitoring.py and polygon.py are in the same directory or Python path
try:
    # Import the modified detector that handles two models
    from monitoring import UnattendedMonitorDetector #
    # Import polygon here at the top level
    import polygon #
except ImportError as e: #
    # Make the error message more specific if possible
    missing_module = "monitoring" if "monitoring" in str(e) else "polygon" if "polygon" in str(e) else "required modules" #
    print(f"Error importing {missing_module}: {e}") #
    print(f"Please ensure {missing_module}.py is in the correct directory or Python path.") #
    sys.exit(1) #

# --- Application Constants ---
WINDOW_NAME_VIEW = "Unattended Monitor Detection (ROI Filtered)" #
# Colors:
ROI_DRAW_COLOR = (255, 0, 0) # Blue for drawing loaded ROIs #
PERSON_COLOR = (255, 0, 0)   # Blue for persons #
ATTENDED_COLOR = (0, 255, 0)   # Green for attended monitors (on) #
UNATTENDED_ON_COLOR = (0, 0, 255) # Red for unattended monitors (on) #
UNATTENDED_OFF_COLOR = (0, 0, 255) # Red for unattended monitors (off) #
THICKNESS = 2 #


def filter_detections_by_roi(all_monitors, all_persons, rois_np_list):
    """
    Filters detections and groups them by the ROI they fall into.
    Args:
        all_monitors (list): List of all detected monitor boxes (may include track_id). # MODIFIED
        all_persons (list): List of all detected person boxes (may include track_id).
        rois_np_list (list): List of ROI polygons as NumPy arrays, or None/empty.
    Returns:
        tuple: (monitors_by_roi, persons_by_roi)
               Dictionaries mapping ROI index {0, 1, ...} to list of objects within it.
               Returns empty dicts if no ROIs are provided.
    """
    monitors_by_roi = defaultdict(list) #
    persons_by_roi = defaultdict(list) #

    if not rois_np_list: #
        return monitors_by_roi, persons_by_roi #

    # Filter monitors
    for m_box in all_monitors: #
        # Use *_ to handle potential extra elements like confidence, track_id
        x1, y1, x2, y2, *_ = m_box #
        center_x = (x1 + x2) / 2 #
        center_y = (y1 + y2) / 2 #
        for i, roi in enumerate(rois_np_list): #
            if cv2.pointPolygonTest(roi, (int(center_x), int(center_y)), False) >= 0: #
                monitors_by_roi[i].append(m_box) #
                break # Monitor belongs to the first ROI it's found in #

    # Filter persons
    for p_box in all_persons: #
        if len(p_box) < 6: continue # Need track_id #

        x1, y1, x2, y2, *rest = p_box #
        center_x = (x1 + x2) / 2 #
        center_y = (y1 + y2) / 2 #
        for i, roi in enumerate(rois_np_list): #
            if cv2.pointPolygonTest(roi, (int(center_x), int(center_y)), False) >= 0: #
                persons_by_roi[i].append(p_box) #
                # Don't break here for persons; they can be in multiple ROIs #

    return monitors_by_roi, persons_by_roi #

# --- MODIFIED process_video function ---
def process_video(input_path, output_path, show_preview, final_rois_np, args):
    """
    Processes the video: reads frames, detects objects, filters by ROI,
    determines monitor status/attendance, annotates, tracks person movements
    between attended ROIs, tracks multi-ROI persons, and saves/shows output.
    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        show_preview (bool): Whether to display the processing preview window.
        final_rois_np (list or None): List of ROI polygons as NumPy arrays.
        args (argparse.Namespace): Parsed command-line arguments (contains model paths).
    """
    # --- Initial Setup (Video Capture, Writer, Detector) ---
    if not os.path.isfile(input_path): #
        raise FileNotFoundError(f"Input video file not found: {input_path}") #
    output_dir = os.path.dirname(output_path) #
    if output_dir and not os.path.exists(output_dir): #
        os.makedirs(output_dir) #

    cap = cv2.VideoCapture(input_path) #
    if not cap.isOpened(): #
        raise IOError(f"Failed to open video file: {input_path}") #

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #
    fps = cap.get(cv2.CAP_PROP_FPS) #
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #

    if fps <= 0: #
        print(f"Warning: Invalid FPS ({fps}) detected from video. Defaulting to 25 FPS.") #
        fps = 25.0 #
    wait_time = int(1000 / fps) if fps > 0 else 40 #

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #
    try:
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_width, frame_height)) #
        if not out.isOpened(): #
                 raise IOError(f"Failed to open VideoWriter for output file: {output_path}") #
    except Exception as e: #
         raise IOError(f"Failed to initialize VideoWriter: {e}") #

    try:
        detector = UnattendedMonitorDetector( #
            monitor_model_path=args.model, #
            person_model_path=args.person_model, #
            confidence_threshold=args.confidence #
        )
    except Exception as e: #
         print(f"Error initializing UnattendedMonitorDetector: {e}") #
         if cap.isOpened(): cap.release() #
         if out.isOpened(): out.release() #
         raise #

    # --- State Tracking Variables ---
    frame_count = 0 #
    unattended_event_frames = 0 #
    max_unattended_in_frame = 0 #
    processing_times = [] #
    person_last_roi = {} # Dictionary to store {track_id: last_roi_index} for movement alert #

    print(f"\n--- Processing Video ---") #
    print(f"Input: {input_path}") #
    print(f"Output: {output_path}") #
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}") #
    print(f"Monitor Model: {args.model or 'Default/Pretrained'}") #
    print(f"Person Model: {args.person_model or 'Default/Pretrained'}") #
    print(f"Confidence Threshold: {args.confidence}") #
    print(f"Using ROIs: {'Yes (' + str(len(final_rois_np)) + ')' if final_rois_np else 'No'}") #
    print(f"Show Preview: {show_preview}") #
    print("------------------------") #

    try:
        while cap.isOpened(): #
            ret, frame = cap.read() #
            if not ret: #
                break #

            frame_count += 1 #
            start_time = time.time() #
            annotated_frame = frame.copy() #
            has_rois = bool(final_rois_np) #

            # Initialize frame-specific trackers
            movement_alerts = [] # Store alert messages for this frame #
            person_roi_locations_this_frame = defaultdict(set) # Tracks {track_id: {roi_index1, roi_index2, ...}} #

            # 1. Draw ROIs
            if has_rois: #
                for i, roi_poly in enumerate(final_rois_np): #
                    cv2.polylines(annotated_frame, [roi_poly], isClosed=True, color=ROI_DRAW_COLOR, thickness=THICKNESS) #
                    # Optional: Label ROIs
                    roi_center = roi_poly.mean(axis=0).astype(int)
                    cv2.putText(annotated_frame, f"ROI {i+1}", (roi_center[0], roi_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_DRAW_COLOR, 2)

            # 2. Detect all objects (monitors now have track_id)
            all_monitors, all_persons = detector.detect_objects(frame) #

            # 3. Filter detections based on ROIs
            monitors_by_roi = defaultdict(list) #
            persons_by_roi = defaultdict(list) #
            if has_rois: #
                monitors_by_roi, persons_by_roi = filter_detections_by_roi( #
                    all_monitors, all_persons, final_rois_np
                )
                # Ensure persons have track ID for subsequent logic
                persons_in_any_roi = [p for roi_list in persons_by_roi.values() for p in roi_list if len(p) >= 6] #
            else: # No ROIs defined #
                monitors_in_any_roi = all_monitors # Use all monitors if no ROIs #
                persons_in_any_roi = [p for p in all_persons if len(p) >= 6] # Use all persons with IDs if no ROIs #


            # 4. Process monitors ROI by ROI & Identify Attending Persons
            unattended_this_frame = [] #
            attending_persons_current_frame = set() # Track IDs of persons attending *any* monitor this frame #

            if has_rois: #
                for roi_index, monitors_in_this_roi in monitors_by_roi.items(): #
                    persons_in_this_roi = persons_by_roi.get(roi_index, []) #
                    for monitor_box in monitors_in_this_roi: #
                        # CHANGE: Unpack including the track_id
                        # Make sure the order matches what detect_objects returns
                        # (x1, y1, x2, y2, confidence, track_id)
                        if len(monitor_box) < 6: continue # Skip if track_id is somehow missing
                        x1, y1, x2, y2, _, monitor_track_id = monitor_box # Ignore confidence for now

                        is_on = detector.is_monitor_on(frame, monitor_box) #

                        if is_on: #
                            attended_by_person_id = None #
                            # Check proximity against persons *in the same ROI*
                            for person_box in persons_in_this_roi: #
                                if len(person_box) < 6: continue # Ensure person has track ID #
                                px1, py1, px2, py2, _, person_track_id = person_box # Get person ID too #

                                # Proximity calculation logic (copied/adapted from check_person_proximity for clarity)
                                prox_radius = max(x2 - x1, y2 - y1) * detector.proximity_radius_multiplier #
                                prox_x1, prox_y1 = max(0, int(x1 - prox_radius)), max(0, int(y1 - prox_radius)) #
                                prox_x2, prox_y2 = int(x2 + prox_radius), int(y2 + prox_radius) #
                                no_overlap = (px2 < prox_x1 or px1 > prox_x2 or py2 < prox_y1 or py1 > prox_y2) #

                                if not no_overlap: #
                                    attended_by_person_id = person_track_id # Store the ID of the attending person #
                                    break # Monitor attended by this person #

                            if attended_by_person_id is not None: #
                                # Monitor is attended
                                attending_persons_current_frame.add(attended_by_person_id) #
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ATTENDED_COLOR, THICKNESS) #
                                # CHANGE: Optionally add Monitor ID to label
                                label_text = f"Attended P:{attended_by_person_id} (M:{monitor_track_id})"
                                cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ATTENDED_COLOR, THICKNESS) #
                            else:
                                # Monitor is ON but Unattended
                                unattended_this_frame.append(monitor_box) #
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_ON_COLOR, THICKNESS) #
                                # CHANGE: Optionally add Monitor ID to label
                                label_text = f"{monitor_track_id} Unattended(On)"
                                cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_ON_COLOR, THICKNESS) #
                        else:
                            # Monitor is OFF (implicitly unattended)
                            unattended_this_frame.append(monitor_box) #
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_OFF_COLOR, THICKNESS) #
                            # CHANGE: Optionally add Monitor ID to label
                            label_text = f"{monitor_track_id} Unattended(Off)"
                            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_OFF_COLOR, THICKNESS) #
            else: # --- Process without ROIs --- #
                 # Use monitors_in_any_roi and persons_in_any_roi here
                 for monitor_box in monitors_in_any_roi: # monitors_in_any_roi now contains track IDs #
                      # CHANGE: Unpack including the track_id
                      if len(monitor_box) < 6: continue # Skip if track_id is somehow missing
                      x1, y1, x2, y2, _, monitor_track_id = monitor_box

                      is_on = detector.is_monitor_on(frame, monitor_box) #
                      if is_on: #
                          # Use the detector's proximity check which takes the full person list
                          is_attended = detector.check_person_proximity(monitor_box, persons_in_any_roi) #
                          if not is_attended: #
                              # Monitor is ON but Unattended
                              unattended_this_frame.append(monitor_box) #
                              cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_ON_COLOR, THICKNESS) #
                              # CHANGE: Optionally add Monitor ID to label
                              label_text = f"{monitor_track_id} Unattended(On)"
                              cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_ON_COLOR, THICKNESS) #
                          else:
                              # Monitor is attended
                              cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ATTENDED_COLOR, THICKNESS) #
                              # CHANGE: Optionally add Monitor ID to label (no person ID available directly here)
                              label_text = f"{monitor_track_id} Attended"
                              cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ATTENDED_COLOR, THICKNESS) #
                      else:
                          # Monitor is OFF (implicitly unattended)
                          unattended_this_frame.append(monitor_box) #
                          cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_OFF_COLOR, THICKNESS) #
                          # CHANGE: Optionally add Monitor ID to label
                          label_text = f"{monitor_track_id} Unattended(Off)"
                          cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_OFF_COLOR, THICKNESS) #


            # 5. Draw ALL persons in ANY ROI, Check Movement & Multi-ROI status
            current_persons_first_roi_map = {} # Map track_id to the *first* ROI index found #
            processed_person_ids_drawing = set() # Avoid drawing same person multiple times #

            if has_rois: #
                 # First pass: Record all locations and draw boxes
                 for roi_index, persons_in_this_roi in persons_by_roi.items(): #
                     for p_box in persons_in_this_roi: #
                         if len(p_box) < 6: continue #
                         px1, py1, px2, py2, _, track_id = p_box #

                         # Record *all* ROIs this person is in
                         person_roi_locations_this_frame[track_id].add(roi_index) #

                         # Store the *first* ROI found for movement logic
                         if track_id not in current_persons_first_roi_map: #
                            current_persons_first_roi_map[track_id] = roi_index #

                         # Draw person box and ID only once per frame
                         if track_id not in processed_person_ids_drawing: #
                             label_text = f"Person {track_id}" #
                             cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), PERSON_COLOR, THICKNESS) #
                             cv2.putText(annotated_frame, label_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, THICKNESS) #
                             processed_person_ids_drawing.add(track_id) #

                 # Second pass: Check for movement alerts
                 for track_id, current_roi in current_persons_first_roi_map.items(): #
                     last_roi = person_last_roi.get(track_id) #
                     # Alert only if person moved AND is currently attending a monitor
                     if last_roi is not None and last_roi != current_roi and track_id in attending_persons_current_frame: #
                          alert = f"ProHance: Person {track_id} Attending in ROI {current_roi + 1}" #
                          movement_alerts.append(alert) #

                 # Update last known ROI state for next frame
                 person_last_roi = current_persons_first_roi_map.copy() #

            else: # Draw persons without ROI logic #
                 # Use persons_in_any_roi which contains only persons with IDs
                 # Create a dictionary to ensure unique drawing per track_id
                 unique_persons_to_draw = {p_box[5]: p_box for p_box in persons_in_any_roi if len(p_box) >=6} #
                 for track_id, p_box in unique_persons_to_draw.items(): #
                    px1, py1, px2, py2, *_ = p_box #
                    label_text = f"Person {track_id}" #
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), PERSON_COLOR, THICKNESS) #
                    cv2.putText(annotated_frame, label_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOR, THICKNESS) #


            # 6. Update general statistics
            # Count only unattended monitors that are ON
            unattended_on_count = sum( #
                1 for box in unattended_this_frame if detector.is_monitor_on(frame, box) #
            )
            if unattended_on_count > 0: #
                 unattended_event_frames += 1 #
                 max_unattended_in_frame = max(max_unattended_in_frame, unattended_on_count) #


            # 7. Add statistics overlay (INCLUDING MOVEMENT & MULTI-ROI ALERTS)
            base_stats_text = f"Unattended Systems: {unattended_on_count}" #
            if has_rois: #
                 base_stats_text += " (ROI)" #

            # Identify persons in multiple ROIs (Only possible if ROIs exist)
            multi_roi_persons = [] #
            multi_roi_alert = [] #
            if has_rois: #
                multi_roi_persons = [ #
                    tid for tid, rois in person_roi_locations_this_frame.items() if len(rois) > 1 #
                ]
                if multi_roi_persons: #
                    ids_str = ", ".join(map(str, sorted(multi_roi_persons))) #
                    multi_roi_alert = [f"Multi-ROI Persons: {ids_str}"] #

            # Combine all alert lines
            combined_stats_lines = [base_stats_text] + movement_alerts + multi_roi_alert #

            # Draw text lines with background
            text_y_start = frame_height - 140 # Adjusted Y position #
            line_height_approx = 20 #

            for i, line in enumerate(combined_stats_lines): #
                 try:
                     (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) #
                     text_x = 10 #
                     text_y = text_y_start + (i * line_height_approx) #
                     if text_y > frame_height - 10: break # Avoid drawing off-screen #

                     # Background rectangle
                     cv2.rectangle(annotated_frame, (text_x - 2, text_y - h - 2), (text_x + w + 2, text_y + 2), (0,0,0), -1) #
                     cv2.putText(annotated_frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text #
                 except Exception as text_e: #
                     print(f"Warning: Error drawing text line '{line}': {text_e}") #


            # 8. Timing, Output, and Preview
            processing_time = time.time() - start_time #
            processing_times.append(processing_time) #
            out.write(annotated_frame) #

            if show_preview: #
                cv2.imshow(WINDOW_NAME_VIEW, annotated_frame) #
                if cv2.waitKey(wait_time) & 0xFF == ord('q'): #
                    print("\nPreview quit by user.") #
                    break #

            # Print progress periodically
            if frame_count % 100 == 0 and total_frames > 0: #
                elapsed_time = sum(processing_times) #
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0 #
                eta_seconds = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0 #
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 0 else "N/A" #
                print(f"Processed: {frame_count}/{total_frames} frames | " #
                      f"Avg FPS: {avg_fps:.2f} | ETA: {eta_str}      ", end='\r') #


    finally:
        # --- Cleanup and Summary ---
        print() # Newline after progress indicator #
        if cap.isOpened(): cap.release() #
        if out.isOpened(): out.release() #
        if show_preview: #
             cv2.waitKey(1) #
             cv2.destroyAllWindows() #
             cv2.waitKey(1) #

        print("\n--- Processing Summary ---") #
        print(f"Total frames processed: {frame_count}") #
        if processing_times: #
             total_time = sum(processing_times) #
             avg_proc_time = total_time / frame_count if frame_count > 0 else 0 #
             avg_fps = 1.0 / avg_proc_time if avg_proc_time > 0 else 0 #
             print(f"Total processing time: {total_time:.2f} seconds") #
             print(f"Average processing time per frame: {avg_proc_time:.4f} seconds") #
             print(f"Average processing FPS: {avg_fps:.2f}") #
        else:
            print("No frames were processed.") #
        print(f"Frames with unattended 'ON' systems detected: {unattended_event_frames}") #
        print(f"Maximum unattended 'ON' systems in a single frame: {max_unattended_in_frame}") #
        print(f"Output video saved to: {output_path}") #
        print("-------------------------") #


# --- main() function ---
def main():
    """Parses arguments, handles ROI definition, and starts video processing."""
    parser = argparse.ArgumentParser(description='Unattended Monitor Detection with YOLOv8 and Optional ROI Filtering.') #
    parser.add_argument('--input', '-i', required=True, help='Path to the input video file.') #
    parser.add_argument('--output', '-o', required=True, help='Path to save the output video file.') #
    parser.add_argument('--model', '-m', default=None, help='Optional path to the YOLO model file (.pt) for MONITOR/GENERAL detection. Uses default if not specified.') #
    parser.add_argument('--person-model', '-p', default=None, help='Optional path to the YOLO model file (.pt) specifically for PERSON detection. Uses default if not specified.') #
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Object detection confidence threshold (0.0 to 1.0). Default: 0.5') #
    parser.add_argument('--no-preview', action='store_true', help='Disable the live preview window during processing.') #
    parser.add_argument('--define-roi', action='store_true', help='Force run the ROI definition process, ignoring the config flag.') #

    args = parser.parse_args() #

    # --- ROI Setup ---
    # Ensure config file exists or create default BEFORE loading
    if not os.path.exists(polygon.CONFIG_FILE): #
        try:
            polygon.create_default_config() #
        except Exception as e: #
             print(f"Error creating default config file: {e}. Cannot proceed without config.") #
             return 1 #

    # Load configuration reliably now that polygon is imported at top
    try:
        define_flag_from_config, saved_rois = polygon.load_config() #
    except Exception as e: #
         print(f"Fatal error loading configuration file ({polygon.CONFIG_FILE}): {e}") #
         return 1 #

    needs_roi_definition = args.define_roi or define_flag_from_config #

    final_rois = [] #
    final_rois_np = None #

    if needs_roi_definition: #
        print("\n--- ROI Definition Required ---") #
        if not os.path.isfile(args.input): #
             print(f"Error: Input video for ROI definition not found: {args.input}") #
             return 1 #
        try:
            # Call definition function which is part of the imported polygon module
            defined_polygons = polygon.define_rois(args.input) #
        except Exception as e: #
            print(f"Error during ROI definition process: {e}") #
            import traceback #
            traceback.print_exc() #
            return 1 #

        if defined_polygons is not None: #
            print(f"ROIs defined and saved: {len(defined_polygons)} zones.") #
            final_rois = defined_polygons #
        else:
            print("ROI definition cancelled or failed. Exiting.") #
            return 1 #
    else:
        print("\n--- Loading Existing ROIs ---") #
        final_rois = saved_rois #
        if not final_rois: #
            print("NOTE: No ROIs found in config.py. Processing entire frame.") #
        elif isinstance(final_rois, list): # Basic check if it's a list #
            print(f"Loaded {len(final_rois)} ROIs from config.py.") #
        else:
            print(f"Warning: ROI_ZONES in {polygon.CONFIG_FILE} is not a list. Processing entire frame.") #
            final_rois = [] # Treat as empty if format is wrong #


    # Convert final ROIs to NumPy format
    if final_rois and isinstance(final_rois, list): #
        try:
            # Add more robust validation before conversion
            valid_polygons = [] #
            for i, poly in enumerate(final_rois): #
                 if not isinstance(poly, list) or len(poly) < 3: #
                     print(f"Warning: Skipping invalid ROI #{i+1} (not a list or < 3 points) in {polygon.CONFIG_FILE}.") #
                     continue #
                 if not all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in poly): #
                     print(f"Warning: Skipping invalid ROI #{i+1} (contains non-point elements) in {polygon.CONFIG_FILE}.") #
                     continue #
                 valid_polygons.append(poly) #

            if valid_polygons: #
                final_rois_np = [np.array(poly, dtype=np.int32) for poly in valid_polygons] #
            else:
                print("Warning: No valid ROIs found after validation. Processing entire frame.") #
                final_rois_np = None #

        except Exception as e: #
             print(f"Error converting ROIs to NumPy arrays: {e}") #
             print(f"Please check the format of ROI_ZONES in {polygon.CONFIG_FILE}.") #
             final_rois_np = None # Process without ROI filtering on error #

    # --- Start Video Processing ---
    try:
        process_video(args.input, args.output, not args.no_preview, final_rois_np, args) #
    except FileNotFoundError as e: #
         print(f"Error: {e}") #
         return 1 #
    except IOError as e: #
         print(f"Input/Output Error: {e}") #
         return 1 #
    except Exception as e: #
        print(f"\nAn unexpected error occurred during video processing:") #
        import traceback #
        traceback.print_exc() #
        return 1 #

    print("\nScript finished successfully.") #
    return 0 #

if __name__ == "__main__": #
    exit_code = main() #
    sys.exit(exit_code) #