# app.py (Complete Modified Version)

import cv2
import numpy as np
import argparse
import os
import time
import sys

# Import components from other modules
# Ensure monitoring.py and polygon.py are in the same directory or Python path
try:
    from monitoring import UnattendedMonitorDetector
    import polygon # Import the polygon module itself
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure monitoring.py and polygon.py are in the correct directory.")
    sys.exit(1)


# --- Application Constants ---
WINDOW_NAME_VIEW = "Unattended Monitor Detection (ROI Filtered)"
# Colors specified by user in the prompt:
ROI_DRAW_COLOR = (255, 0, 0) # Blue for drawing loaded ROIs (line is commented out below)
PERSON_COLOR = (255, 0, 0)   # Blue for persons
ATTENDED_COLOR = (0, 255, 0)   # Green for attended monitors (on)
UNATTENDED_ON_COLOR = (0, 0, 255) # Red for unattended monitors (on)
UNATTENDED_OFF_COLOR = (0, 0, 255) # Red for unattended monitors (off) - User specified red for both
THICKNESS = 2


def filter_detections_by_roi(all_monitors, all_persons, rois_np_list):
    """
    Filters detections and groups them by the ROI they fall into.

    Args:
        all_monitors (list): List of all detected monitor boxes.
        all_persons (list): List of all detected person boxes.
        rois_np_list (list): List of ROI polygons as NumPy arrays, or None/empty.

    Returns:
        tuple: (monitors_by_roi, persons_by_roi)
               Dictionaries mapping ROI index to list of objects within it.
               Returns empty dicts if no ROIs are provided.
    """
    monitors_by_roi = {}
    persons_by_roi = {}

    if not rois_np_list: # Handle case with no ROIs
        # If no ROIs, we could treat the whole frame as one ROI,
        # but for clarity, let's return empty dicts and handle downstream.
        # Or, assign all detections to a special key like -1?
        # Let's keep it simple: if no ROIs, filtering results in nothing *grouped by ROI*.
        # The calling function (process_video) will need to handle this.
        return monitors_by_roi, persons_by_roi

    # Initialize dictionaries with empty lists for each ROI index
    for i in range(len(rois_np_list)):
        monitors_by_roi[i] = []
        persons_by_roi[i] = []

    # Filter monitors and assign to the correct ROI index
    for m_box in all_monitors:
        # Use center point for checking containment
        x1, y1, x2, y2, _ = m_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        for i, roi in enumerate(rois_np_list):
            # pointPolygonTest returns +ve if inside, 0 if on boundary, -ve if outside
            if cv2.pointPolygonTest(roi, (int(center_x), int(center_y)), False) >= 0:
                monitors_by_roi[i].append(m_box)
                break # Found in one ROI, stop checking others for this monitor

    # Filter persons and assign to the correct ROI index
    for p_box in all_persons:
        # Use center point for checking containment
        x1, y1, x2, y2, _ = p_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        for i, roi in enumerate(rois_np_list):
            if cv2.pointPolygonTest(roi, (int(center_x), int(center_y)), False) >= 0:
                persons_by_roi[i].append(p_box)
                break # Found in one ROI, stop checking others for this person

    return monitors_by_roi, persons_by_roi


def process_video(input_path, output_path, show_preview, final_rois_np, args):
    """
    Processes the video: reads frames, detects objects, filters by ROI,
    determines monitor status/attendance, annotates, and saves/shows output.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        show_preview (bool): Whether to display the processing preview window.
        final_rois_np (list or None): List of ROI polygons as NumPy arrays.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate FPS and calculate wait time
    if fps <= 0:
        print(f"Warning: Invalid FPS ({fps}) detected from video. Defaulting to 25 FPS.")
        fps = 25.0 # Use a float for potentially more accurate writer init
    wait_time = int(1000 / fps) if fps > 0 else 40 # Default to 40ms (25fps) if calculation fails

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard MP4 codec
    try:
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_width, frame_height))
        if not out.isOpened():
             raise IOError(f"Failed to open VideoWriter for output file: {output_path}")
    except Exception as e:
         raise IOError(f"Failed to initialize VideoWriter: {e}")


    # Initialize the detector using args for customization
    try:
        detector = UnattendedMonitorDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
    except Exception as e:
         print(f"Error initializing UnattendedMonitorDetector: {e}")
         cap.release() # Release capture if detector fails
         out.release() # Release writer
         raise # Re-raise the exception

    # Statistics tracking
    frame_count = 0
    unattended_event_frames = 0 # Frames where at least one monitor was unattended
    max_unattended_in_frame = 0 # Max number of unattended monitors in a single frame
    processing_times = []

    print(f"\n--- Processing Video ---")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps:.2f} (Wait: {wait_time}ms)")
    print(f"Total Frames: ~{total_frames}" if total_frames > 0 else "Unknown")
    if final_rois_np:
        print(f"ROIs Active: Yes ({len(final_rois_np)} zones)")
    else:
        print("ROIs Active: No (Processing full frame)")
    print("------------------------")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                #print("\nEnd of video or cannot read frame.") # Reduce verbosity
                break

            frame_count += 1
            start_time = time.time()
            annotated_frame = frame.copy() # Work on a copy for annotations

            # --- ROI Handling ---
            has_rois = bool(final_rois_np) # True if ROIs are defined and loaded

            # 1. Draw defined ROIs onto the frame first (OPTIONAL)
            #    This line is commented out as per previous request
            # if has_rois:
            #     # cv2.polylines(annotated_frame, final_rois_np, isClosed=True, color=ROI_DRAW_COLOR, thickness=THICKNESS)
            #     pass

            # 2. Detect all objects in the frame
            all_monitors, all_persons = detector.detect_objects(frame) # Use original frame for detection

            # 3. Filter detections based on ROIs (if ROIs exist)
            monitors_by_roi = {}
            persons_by_roi = {}
            if has_rois:
                monitors_by_roi, persons_by_roi = filter_detections_by_roi(
                    all_monitors, all_persons, final_rois_np
                )
                # Get flat lists of unique objects within any ROI for drawing/stats later
                monitors_in_any_roi = [m for roi_list in monitors_by_roi.values() for m in roi_list]
                persons_in_any_roi = [p for roi_list in persons_by_roi.values() for p in roi_list]
            else:
                # If no ROIs, treat all detected objects as relevant
                monitors_in_any_roi = all_monitors
                persons_in_any_roi = all_persons

            # 4. Process monitors ROI by ROI (if ROIs exist) or all monitors (if no ROIs)
            unattended_this_frame = [] # List to store boxes of unattended monitors in this frame
            total_monitors_processed = 0
            total_persons_processed = len(persons_in_any_roi) # Count persons once

            if has_rois:
                # Iterate through each ROI index that had monitors
                for roi_index in monitors_by_roi:
                    monitors_in_this_roi = monitors_by_roi[roi_index]
                    persons_in_this_roi = persons_by_roi.get(roi_index, []) # Persons in this specific ROI

                    total_monitors_processed += len(monitors_in_this_roi)

                    # Process each monitor found specifically in this ROI
                    for monitor_box in monitors_in_this_roi:
                        x1, y1, x2, y2, _ = monitor_box
                        is_on = detector.is_monitor_on(frame, monitor_box)

                        if is_on:
                            # Check proximity using ONLY persons from the SAME ROI
                            is_attended = detector.check_person_proximity(monitor_box, persons_in_this_roi)

                            if not is_attended:
                                unattended_this_frame.append(monitor_box)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_ON_COLOR, THICKNESS)
                                cv2.putText(annotated_frame, "Unattended(On)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_ON_COLOR, THICKNESS)
                            else:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ATTENDED_COLOR, THICKNESS)
                                cv2.putText(annotated_frame, "Attended", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ATTENDED_COLOR, THICKNESS)
                        else:
                            # Monitor is off (unattended if inside ROI)
                            unattended_this_frame.append(monitor_box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_OFF_COLOR, THICKNESS)
                            cv2.putText(annotated_frame, "Unattended(Off)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_OFF_COLOR, THICKNESS)
            else:
                # No ROIs - process all detected monitors
                total_monitors_processed = len(all_monitors)
                for monitor_box in all_monitors:
                    x1, y1, x2, y2, _ = monitor_box
                    is_on = detector.is_monitor_on(frame, monitor_box)
                    if is_on:
                        # Check proximity using ALL detected persons if no ROIs
                        is_attended = detector.check_person_proximity(monitor_box, all_persons)
                        if not is_attended:
                            unattended_this_frame.append(monitor_box)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_ON_COLOR, THICKNESS)
                            cv2.putText(annotated_frame, "Unattended(On)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_ON_COLOR, THICKNESS)
                        else:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ATTENDED_COLOR, THICKNESS)
                            cv2.putText(annotated_frame, "Attended", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ATTENDED_COLOR, THICKNESS)
                    else:
                        unattended_this_frame.append(monitor_box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_OFF_COLOR, THICKNESS)
                        cv2.putText(annotated_frame, "Unattended(Off)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_OFF_COLOR, THICKNESS)


            # 5. Draw boxes for ALL persons found within ANY ROI (or all persons if no ROIs)
            #    Use set to handle potential duplicates if ROIs overlap and a person is in multiple
            unique_persons_to_draw = {tuple(p_box[:4]): p_box for p_box in persons_in_any_roi} # Use coords as key
            for p_box in unique_persons_to_draw.values():
                 px1, py1, px2, py2, _ = p_box
                 cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), PERSON_COLOR, THICKNESS)


            # 6. Update statistics based on this frame's results
            # if unattended_this_frame:
            #     unattended_event_frames += 1
            #     max_unattended_in_frame = max(max_unattended_in_frame, len(unattended_this_frame))
            unattended_on_count = sum(
                1 for box in unattended_this_frame if detector.is_monitor_on(frame, box)
            )

            # 7. Add statistics overlay
            #    Counts reflect objects processed (either all or within ROIs)
            # stats_text = f"Unattended Systems: {len(unattended_this_frame)}"
            stats_text = f"Unattended Systems: {unattended_on_count}"
            if has_rois:
                 stats_text += "" # Indicate stats are ROI-based
            # Add background rectangle for better visibility
            try: # Wrap text drawing in try-except for robustness
                (w, h), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = 70
                text_y = frame_height - 120
                cv2.rectangle(annotated_frame, (text_x - 5, text_y - h - 5), (text_x + w + 5, text_y + 5), (0,0,0), -1) # Background
                cv2.putText(annotated_frame, stats_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text
            except Exception as text_e:
                 print(f"Warning: Error drawing text overlay: {text_e}")


            # 8. Timing, Output, and Preview
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            out.write(annotated_frame) # Write the annotated frame

            if show_preview:
                cv2.imshow(WINDOW_NAME_VIEW, annotated_frame)
                # Use calculated wait_time, allow quitting with 'q'
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    print("\nPreview quit by user.")
                    break

            # Print progress periodically
            if total_frames > 0 and frame_count % 50 == 0:
                 progress = frame_count / total_frames * 100
                 avg_proc_time_so_far = sum(processing_times) / len(processing_times) if processing_times else 0
                 current_fps = 1.0 / avg_proc_time_so_far if avg_proc_time_so_far > 0 else 0
                 print(f"\rProcessed {frame_count}/{total_frames} frames ({progress:.1f}%) - Current Proc FPS: {current_fps:.2f}", end="")
            elif frame_count % 50 == 0: # Case where total_frames is unknown
                  avg_proc_time_so_far = sum(processing_times) / len(processing_times) if processing_times else 0
                  current_fps = 1.0 / avg_proc_time_so_far if avg_proc_time_so_far > 0 else 0
                  print(f"\rProcessed {frame_count} frames - Current Proc FPS: {current_fps:.2f}", end="")


    finally:
        # Ensure final newline after progress printing
        print()
        # Release all resources
        if cap.isOpened(): cap.release()
        if out.isOpened(): out.release()
        if show_preview:
             # Add small delay before destroying windows sometimes helps avoid glitches
             cv2.waitKey(1)
             cv2.destroyAllWindows()
             cv2.waitKey(1) # Extra waitKey can sometimes help ensure window closes

    # Print final summary statistics
    print("\n--- Processing Summary ---")
    if frame_count > 0 and processing_times: # Ensure times were recorded
        avg_processing_time = sum(processing_times) / frame_count
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        unattended_percentage = (unattended_event_frames / frame_count) * 100

        print(f"Total frames processed : {frame_count}")
        print(f"Avg. processing time : {avg_processing_time:.4f} sec/frame")
        print(f"Avg. processing speed: {avg_fps:.2f} FPS")
        stats_scope = "(in ROI)" if has_rois else "(Full Frame)"
        print(f"Frames with unattended monitors {stats_scope}: {unattended_event_frames} ({unattended_percentage:.1f}%)")
        print(f"Max simultaneous unattended monitors {stats_scope}: {max_unattended_in_frame}")
    else:
        print("No frames were processed or processing times recorded.")
    print(f"Output video saved to: {output_path}")
    print("-------------------------")


def main():
    """Parses arguments, handles ROI definition, and starts video processing."""
    parser = argparse.ArgumentParser(description='Unattended Monitor Detection with YOLOv8 and Optional ROI Filtering.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input video file.')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output video file.')
    parser.add_argument('--model', '-m', default=None, help='Optional path to custom YOLO model file (.pt). Uses default/pretrained if not specified.')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Object detection confidence threshold (0.0 to 1.0). Default: 0.5')
    parser.add_argument('--no-preview', action='store_true', help='Disable the live preview window during processing.')
    parser.add_argument('--define-roi', action='store_true', help='Force run the ROI definition process, ignoring the config flag.')

    args = parser.parse_args()

    # --- ROI Setup ---
    # Ensure config file exists (or is created with default DEFINE_ROI=True)
    # create_default_config might be better placed inside load_config if file not found
    if not os.path.exists(polygon.CONFIG_FILE):
         polygon.create_default_config()

    try:
        define_flag_from_config, saved_rois = polygon.load_config()
    except Exception as e:
         print(f"Fatal error loading configuration file ({polygon.CONFIG_FILE}): {e}")
         return 1 # Exit if config is critically broken

    # Determine if ROI definition is needed
    needs_roi_definition = args.define_roi or define_flag_from_config

    final_rois = []
    final_rois_np = None # Initialize as None to signify no ROIs loaded/defined yet

    if needs_roi_definition:
        print("\n--- ROI Definition Required ---")
        if not os.path.isfile(args.input):
             print(f"Error: Input video for ROI definition not found: {args.input}")
             return 1

        # Run the definition process from the polygon module
        defined_polygons = polygon.define_rois(args.input)

        if defined_polygons is not None:
            # ROIs were defined and saved by polygon.define_rois
            print(f"ROIs defined and saved: {len(defined_polygons)} zones.")
            final_rois = defined_polygons # Use the newly defined ROIs for this run
        else:
            # User quit ('q') or an error occurred during definition
            print("ROI definition cancelled or failed. Exiting.")
            return 1 # Exit if definition was required but not completed
    else:
        print("\n--- Loading Existing ROIs ---")
        final_rois = saved_rois # Use ROIs loaded from config
        if not final_rois:
            print("NOTE: No ROIs found in config.py. Processing entire frame.")
        else:
            print(f"Loaded {len(final_rois)} ROIs from config.py.")

    # Convert the final list of ROI coordinates to NumPy arrays for OpenCV
    # This happens whether ROIs were just defined or loaded
    if final_rois:
        try:
            # Basic validation on the structure
            if not isinstance(final_rois, list):
                raise ValueError("ROI_ZONES is not a list.")
            for poly in final_rois:
                 if not isinstance(poly, list) or not all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in poly):
                     raise ValueError("Invalid ROI format in list: Each ROI must be a list of (x, y) points.")
                 if len(poly) < 3:
                      raise ValueError("Invalid ROI: Polygons must have at least 3 points.")
            # Conversion
            final_rois_np = [np.array(poly, dtype=np.int32) for poly in final_rois]
            if not final_rois_np: # Check if list is empty after conversion (e.g., if final_rois was [[]])
                 print("Warning: ROI list was non-empty but resulted in zero valid NumPy ROIs.")
                 final_rois_np = None

        except Exception as e:
             print(f"Error converting loaded/defined ROIs to NumPy arrays: {e}")
             print(f"Please check the format of ROI_ZONES in {polygon.CONFIG_FILE}.")
             print("Processing will continue without ROI filtering.")
             final_rois_np = None # Ensure it's None if conversion fails


    # --- Start Video Processing ---
    try:
        process_video(args.input, args.output, not args.no_preview, final_rois_np, args)
    except FileNotFoundError as e:
         print(f"Error: {e}")
         return 1
    except IOError as e: # Covers VideoCapture/VideoWriter errors
          print(f"Input/Output Error: {e}")
          return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred during video processing:")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return 1

    print("\nScript finished successfully.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)