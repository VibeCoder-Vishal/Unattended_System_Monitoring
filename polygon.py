# polygon.py
import cv2
import numpy as np
import os
import sys
import importlib # To reload config if needed
import time # Import time for potential fallback delay

# --- Configuration ---
CONFIG_FILE = 'config.py'

# VIDEO_SOURCE = 0 # Use 0 for webcam
VIDEO_SOURCE = "Unattended System/20240828_20240828145744_20240828145834_145745.mp4" # <<<--- PUT YOUR VIDEO FILE PATH HERE


WINDOW_NAME_DEFINE = "Define ROIs - Left Click: Add Point, Right Click: Next Polygon, 's': Save & Exit, 'r': Reset Current, 'c': Clear All, 'q': Quit"
WINDOW_NAME_VIEW = "Video Feed with ROIs"
ROI_COLOR = (0, 255, 0)  # Green
TEMP_LINE_COLOR = (0, 0, 255) # Red
THICKNESS = 2

# --- Global variables for ROI definition ---
current_points = []
polygons = []
frame_copy = None # To draw on without modifying the original frame during definition

# --- Function to load configuration ---
def load_config():
    """Loads configuration from config.py"""
    try:
        # Ensure we get the latest version if the file was modified
        if 'config' in sys.modules:
            importlib.reload(sys.modules['config'])
        import config
        return config.DEFINE_ROI, config.ROI_ZONES
    except ImportError:
        print(f"Error: {CONFIG_FILE} not found. Please create it.")
        # Create a default config if it doesn't exist
        create_default_config()
        import config
        return config.DEFINE_ROI, config.ROI_ZONES
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

# --- Function to create a default config file ---
def create_default_config():
    """Creates a default config.py file if it doesn't exist."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Creating default {CONFIG_FILE}...")
        with open(CONFIG_FILE, 'w') as f:
            f.write("# config.py\n\n")
            f.write("# Set to True to run the ROI definition process on the next run.\n")
            f.write("# Set to False to use the existing ROI_ZONES and skip definition.\n")
            f.write("DEFINE_ROI = True\n\n")
            f.write("# List to store the coordinates of the polygonal ROI zones.\n")
            f.write("# Each zone is a list of (x, y) tuples.\n")
            f.write("# Example: ROI_ZONES = [ [(100, 100), (200, 100), (200, 200), (100, 200)], [(300, 300), (400, 350), (350, 450)] ]\n")
            f.write("ROI_ZONES = []\n")
        print(f"{CONFIG_FILE} created.")

# --- Function to save configuration ---
def save_config(roi_zones_to_save):
    """Saves the defined ROIs to config.py and sets DEFINE_ROI to False."""
    print("Saving ROIs to config.py...")
    try:
        # Read existing content (optional, but safer if config has other settings)
        # For simplicity here, we just overwrite the relevant parts
        with open(CONFIG_FILE, 'w') as f:
            f.write("# config.py\n\n")
            f.write("# Set to True to run the ROI definition process on the next run.\n")
            f.write("# Set to False to use the existing ROI_ZONES and skip definition.\n")
            f.write("DEFINE_ROI = False # Set to False after defining ROIs\n\n") # IMPORTANT: Set to False
            f.write("# List to store the coordinates of the polygonal ROI zones.\n")
            f.write("# Each zone is a list of (x, y) tuples.\n")
            f.write(f"ROI_ZONES = {roi_zones_to_save}\n") # Write the actual data
        print("Configuration saved.")
    except Exception as e:
        print(f"Error saving config: {e}")

# --- Mouse callback function for ROI definition ---
def draw_polygon(event, x, y, flags, param):
    global current_points, polygons, frame_copy

    if frame_copy is None: # Ensure frame_copy is available
        return

    display_frame = frame_copy.copy() # Work on a copy for drawing previews

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"Added point: ({x}, {y}). Current polygon points: {len(current_points)}")
        # Draw existing points for the current polygon
        for point in current_points:
            cv2.circle(display_frame, point, 5, TEMP_LINE_COLOR, -1)
        # Draw lines connecting points as they are added
        if len(current_points) > 1:
            cv2.polylines(display_frame, [np.array(current_points)], isClosed=False, color=TEMP_LINE_COLOR, thickness=THICKNESS)
        cv2.imshow(WINDOW_NAME_DEFINE, display_frame)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3:
            polygons.append(list(current_points)) # Save completed polygon (as list)
            print(f"Polygon {len(polygons)} completed with {len(current_points)} points.")
            current_points = [] # Reset for the next polygon
            # Redraw all completed polygons on the frame_copy for next display
            frame_copy_updated = frame_copy.copy()
            for poly in polygons:
                 cv2.polylines(frame_copy_updated, [np.array(poly)], isClosed=True, color=ROI_COLOR, thickness=THICKNESS)
            cv2.imshow(WINDOW_NAME_DEFINE, frame_copy_updated)

        else:
            print("Need at least 3 points to define a polygon. Right-click ignored.")

# --- Function to define ROIs ---
def define_rois(capture_source):
    global current_points, polygons, frame_copy

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {capture_source}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source.")
        cap.release()
        return None

    frame_copy = frame.copy() # Store the first frame to draw on
    polygons = [] # Reset polygon list for this session
    current_points = [] # Reset current points

    cv2.namedWindow(WINDOW_NAME_DEFINE)
    cv2.setMouseCallback(WINDOW_NAME_DEFINE, draw_polygon)

    print("\n--- ROI Definition ---")
    print("Left Mouse Button: Click to add points for the current polygon.")
    print("Right Mouse Button: Click to finish the current polygon (needs >= 3 points).")
    print("Key 's': Save all defined polygons and exit definition mode.")
    print("Key 'r': Reset the points for the *current* polygon being drawn.")
    print("Key 'c': Clear *all* defined polygons and start over.")
    print("Key 'q': Quit definition mode *without saving*.")
    print("----------------------\n")


    while True:
        display_frame = frame_copy.copy()

        # Draw already completed polygons
        if polygons:
            for poly in polygons:
                 cv2.polylines(display_frame, [np.array(poly)], isClosed=True, color=ROI_COLOR, thickness=THICKNESS)

        # Draw the current polygon being built
        if len(current_points) > 0:
            for point in current_points:
                cv2.circle(display_frame, point, 5, TEMP_LINE_COLOR, -1)
            if len(current_points) > 1:
                cv2.polylines(display_frame, [np.array(current_points)], isClosed=False, color=TEMP_LINE_COLOR, thickness=THICKNESS)

        cv2.imshow(WINDOW_NAME_DEFINE, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'): # Save
            if not polygons:
                print("No polygons defined to save.")
            else:
                save_config(polygons) # Save the list of polygons
                break # Exit definition loop

        elif key == ord('r'): # Reset current polygon points
             print("Resetting points for the current polygon.")
             current_points = []
             # Need to redraw the frame without the points just cleared
             display_frame_reset = frame_copy.copy()
             if polygons:
                for poly in polygons:
                     cv2.polylines(display_frame_reset, [np.array(poly)], isClosed=True, color=ROI_COLOR, thickness=THICKNESS)
             cv2.imshow(WINDOW_NAME_DEFINE, display_frame_reset)

        elif key == ord('c'): # Clear all polygons
            print("Clearing all defined polygons.")
            polygons = []
            current_points = []
            frame_copy = frame.copy() # Restore original clean frame
            cv2.imshow(WINDOW_NAME_DEFINE, frame_copy)

        elif key == ord('q'): # Quit without saving
            print("Quitting ROI definition without saving.")
            polygons = None # Indicate cancellation
            break # Exit definition loop

    # Don't release cap here if using a video file,
    # as we need it for the main loop. Release only frame-grab specific resources.
    cv2.destroyWindow(WINDOW_NAME_DEFINE)
    # If cap was opened just for the first frame, release it.
    # But if VIDEO_SOURCE is a file path, we reuse it later.
    if isinstance(capture_source, int): # If it was a webcam (index)
         cap.release()
    # If it was a file path, keep 'cap' open conceptually,
    # it will be reopened in the main loop section.

    return polygons # Return the defined polygons (or None if cancelled)


# --- Main application ---
if __name__ == "__main__":
    create_default_config() # Create config if it doesn't exist
    define_flag, saved_rois = load_config()

    final_rois = []

    # --- Define ROIs if needed ---
    if define_flag:
        print("Starting ROI definition mode...")
        # Pass the *original* VIDEO_SOURCE to define_rois
        newly_defined_rois = define_rois(VIDEO_SOURCE)
        if newly_defined_rois is not None:
            final_rois = newly_defined_rois
            print(f"ROIs defined: {len(final_rois)} zones.")
            # Config saving happens within define_rois on 's' key press
            # Need to reload config to get the updated DEFINE_ROI=False status
            define_flag, final_rois = load_config() # Reload config state
        else:
            print("ROI definition cancelled or failed. Exiting.")
            sys.exit(0)
    else:
        print("Skipping ROI definition (DEFINE_ROI=False in config.py).")
        final_rois = saved_rois
        if not final_rois:
             print("Warning: DEFINE_ROI is False, but no ROIs found in config.py. You may want to set DEFINE_ROI=True to define them.")
        else:
             print(f"Loaded {len(final_rois)} ROIs from config.py.")


    # --- Main video processing loop (using the defined ROIs) ---
    if final_rois: # Proceed only if we have ROIs (either newly defined or loaded)
        print("\nStarting video feed processing with ROIs...")
        cap = cv2.VideoCapture(VIDEO_SOURCE) # Open the video source AGAIN for playback
        if not cap.isOpened():
            print(f"Error: Could not open video source for processing: {VIDEO_SOURCE}")
            sys.exit(1)

        # --- Calculate Wait Time for Normal Speed ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = 0
        if fps > 0:
            wait_time = int(1000 / fps)
            print(f"Video FPS: {fps:.2f}, Calculated Wait Time: {wait_time} ms")
        else:
            wait_time = 33 # Default to ~30 FPS if FPS is not available
            print(f"Could not get video FPS. Defaulting wait time to {wait_time} ms (~30 FPS)")
        # --- End Calculate Wait Time ---

        while True:
            ret, frame = cap.read()
            if not ret:
                # The HEVC errors often precede this point when frames can't be decoded
                print("End of video or error reading frame.")
                break

            # Draw the final ROIs on the current frame
            if final_rois:
                 # Convert to numpy arrays for cv2.polylines
                np_rois = [np.array(poly, dtype=np.int32) for poly in final_rois]
                cv2.polylines(frame, np_rois, isClosed=True, color=ROI_COLOR, thickness=THICKNESS)

            # --- ADD YOUR VIDEO PROCESSING LOGIC HERE ---
            # ---------------------------------------------

            cv2.imshow(WINDOW_NAME_VIEW, frame)

            # --- Use calculated wait_time ---
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                print("Exiting video feed.")
                break
            # --- End Use calculated wait_time ---

        cap.release()
        cv2.destroyAllWindows()
    else:
         print("No ROIs available to process the video feed.")

    print("Script finished.")