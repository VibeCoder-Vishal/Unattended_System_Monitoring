# app.py

import cv2
import numpy as np
import argparse
import os
import time
import sys

# Import components from other modules
from monitoring import UnattendedMonitorDetector
import polygon # Import the polygon module itself

# --- Application Constants ---
WINDOW_NAME_VIEW = "Unattended Monitor Detection (ROI Filtered)"
ROI_DRAW_COLOR = (255, 0, 0) # Blue for drawing loaded ROIs
PERSON_COLOR = (255, 0, 0)   # Blue for persons
ATTENDED_COLOR = (0, 255, 0)   # Green for attended monitors (on)
UNATTENDED_ON_COLOR = (0, 0, 255) # Red for unattended monitors (on)
UNATTENDED_OFF_COLOR = (0, 0, 255) # Red for unattended monitors (off)
THICKNESS = 2


def filter_detections_by_roi(all_monitors, all_persons, rois_np_list):
    """
    Filters lists of monitors and persons, keeping only those whose
    center point falls within any of the provided ROI polygons.

    Args:
        all_monitors (list): List of all detected monitor boxes.
        all_persons (list): List of all detected person boxes.
        rois_np_list (list): List of ROI polygons as NumPy arrays, or None/empty.

    Returns:
        tuple: (monitors_in_roi, persons_in_roi)
    """
    monitors_in_roi = []
    persons_in_roi = []

    # If no ROIs are defined or provided, return all detections
    if not rois_np_list:
        return all_monitors, all_persons

    # Filter monitors
    for m_box in all_monitors:
        x1, y1, x2, y2, _ = m_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        for roi in rois_np_list:
            # pointPolygonTest returns +ve if inside, 0 if on boundary, -ve if outside
            if cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0:
                monitors_in_roi.append(m_box)
                break # Stop checking other ROIs for this monitor

    # Filter persons
    for p_box in all_persons:
        x1, y1, x2, y2, _ = p_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        for roi in rois_np_list:
            if cv2.pointPolygonTest(roi, (center_x, center_y), False) >= 0:
                persons_in_roi.append(p_box)
                break # Stop checking other ROIs for this person

    return monitors_in_roi, persons_in_roi


def process_video(input_path, output_path, show_preview, final_rois_np, args):
    """
    Processes the video: reads frames, detects objects, filters by ROI,
    determines monitor status/attendance, annotates, and saves/shows output.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the output video.
        show_preview (bool): Whether to display the processing preview window.
        final_rois_np (list): List of ROI polygons as NumPy arrays, or None.
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
    wait_time = int(1000 / fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (frame_width, frame_height)) # Use float fps

    # Initialize the detector using args for customization
    detector = UnattendedMonitorDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )

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
                print("\nEnd of video or cannot read frame.")
                break

            frame_count += 1
            start_time = time.time()
            annotated_frame = frame.copy() # Work on a copy for annotations

            # 1. Draw defined ROIs onto the frame first
            if final_rois_np:
                cv2.polylines(annotated_frame, final_rois_np, isClosed=True, color=ROI_DRAW_COLOR, thickness=THICKNESS)

            # 2. Detect all objects in the frame
            all_monitors, all_persons = detector.detect_objects(frame) # Use original frame for detection

            # 3. Filter detections based on ROIs
            monitors_in_roi, persons_in_roi = filter_detections_by_roi(
                all_monitors, all_persons, final_rois_np
            )

            # 4. Process monitors *within* ROIs
            unattended_this_frame = [] # List to store boxes of unattended monitors in this frame
            total_monitors_on_in_roi = 0

            for monitor_box in monitors_in_roi:
                x1, y1, x2, y2, _ = monitor_box
                is_on = detector.is_monitor_on(frame, monitor_box) # Check status on original frame

                if is_on:
                    total_monitors_on_in_roi += 1
                    # Check proximity using persons *also within ROIs*
                    is_attended = detector.check_person_proximity(monitor_box, persons_in_roi)

                    if not is_attended:
                        unattended_this_frame.append(monitor_box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_ON_COLOR, THICKNESS)
                        cv2.putText(annotated_frame, "Unattended(On)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_ON_COLOR, THICKNESS)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ATTENDED_COLOR, THICKNESS)
                        cv2.putText(annotated_frame, "Attended", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ATTENDED_COLOR, THICKNESS)
                else:
                    # Monitor is off (considered unattended if inside ROI)
                    unattended_this_frame.append(monitor_box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), UNATTENDED_OFF_COLOR, THICKNESS)
                    cv2.putText(annotated_frame, "Unattended(Off)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UNATTENDED_OFF_COLOR, THICKNESS)

            # 5. Draw boxes for persons *within* ROIs
            for p_box in persons_in_roi:
                px1, py1, px2, py2, _ = p_box
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), PERSON_COLOR, THICKNESS)

            # 6. Update statistics based on this frame's results (within ROI)
            if unattended_this_frame:
                unattended_event_frames += 1
                max_unattended_in_frame = max(max_unattended_in_frame, len(unattended_this_frame))

            # # 7. Add statistics overlay (counts based on ROI-filtered detections)
            # stats_text = f"Unattended: {len(unattended_this_frame)} | Monitors(ROI): {len(monitors_in_roi)} | Persons(ROI): {len(persons_in_roi)}"
            # # Add background rectangle for better visibility
            # (w, h), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # cv2.rectangle(annotated_frame, (5, 5), (15 + w, 35), (0,0,0), -1) # Black background
            # cv2.putText(annotated_frame, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text

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
        cap.release()
        out.release()
        if show_preview: # Only destroy window if it was created
             cv2.destroyAllWindows()

    # Print final summary statistics
    print("\n--- Processing Summary ---")
    if frame_count > 0:
        avg_processing_time = sum(processing_times) / frame_count
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        unattended_percentage = (unattended_event_frames / frame_count) * 100

        print(f"Total frames processed : {frame_count}")
        print(f"Avg. processing time : {avg_processing_time:.4f} sec/frame")
        print(f"Avg. processing speed: {avg_fps:.2f} FPS")
        print(f"Frames with unattended monitors (in ROI): {unattended_event_frames} ({unattended_percentage:.1f}%)")
        print(f"Max simultaneous unattended monitors (in ROI): {max_unattended_in_frame}")
    else:
        print("No frames were processed.")
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
    polygon.create_default_config() # Ensure config file exists

    try:
        define_flag_from_config, saved_rois = polygon.load_config()
    except Exception as e:
         print(f"Fatal error loading configuration file ({polygon.CONFIG_FILE}): {e}")
         return 1 # Exit if config is critically broken

    # Determine if ROI definition is needed
    # Force definition if --define-roi flag is used, OR if flag in config is True
    needs_roi_definition = args.define_roi or define_flag_from_config

    final_rois = []
    final_rois_np = None # Initialize as None

    if needs_roi_definition:
        print("\n--- ROI Definition Required ---")
        if not os.path.isfile(args.input):
             print(f"Error: Input video for ROI definition not found: {args.input}")
             return 1

        # Run the definition process from the polygon module
        defined_polygons = polygon.define_rois(args.input)

        if defined_polygons is not None:
            # ROIs were defined and saved (save_config called within define_rois)
            print(f"ROIs defined and saved: {len(defined_polygons)} zones.")
            final_rois = defined_polygons # Use the newly defined ROIs directly
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
    if final_rois:
        try:
            # Validate structure before converting (optional but good practice)
            for poly in final_rois:
                 if not isinstance(poly, list) or not all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in poly):
                     raise ValueError("Invalid ROI format in list.")
            final_rois_np = [np.array(poly, dtype=np.int32) for poly in final_rois]
        except Exception as e:
             print(f"Error converting loaded/defined ROIs to NumPy arrays: {e}")
             print(f"Please check the format of ROI_ZONES in {polygon.CONFIG_FILE} or the output of definition.")
             print("Processing will continue without ROI filtering.")
             final_rois_np = None # Ensure it's None if conversion fails
    # --- End ROI Setup ---


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