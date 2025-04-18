import cv2
import numpy as np
import os
import sys
import importlib
import importlib.util # For more robust import/reload

# --- Constants for ROI Definition ---
CONFIG_FILE = 'config.py'
WINDOW_NAME_DEFINE = "Define ROIs - Left Click: Add Point, Right Click: Next Polygon, 's': Save & Exit, 'r': Reset Current, 'c': Clear All, 'q': Quit"
ROI_DRAW_COLOR = (255, 255, 0) # Cyan for drawing defined polygons during definition
TEMP_LINE_COLOR = (0, 0, 255)  # Red for drawing temporary lines
THICKNESS = 2

# --- Globals variables specific to the drawing process ---
# These track the state *during* the define_rois execution
current_points = []
polygons = []
frame_copy_for_drawing = None

# --- ROI Management Functions ---

def load_config():
    """Loads configuration from config.py"""
    if not os.path.exists(CONFIG_FILE):
         print(f"Warning: {CONFIG_FILE} not found. Creating default.")
         create_default_config()

    try:
        # Use importlib for robust reloading if module already imported
        spec = importlib.util.spec_from_file_location("config", CONFIG_FILE)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config) # Load module from spec
        return config.DEFINE_ROI, config.ROI_ZONES
    except ImportError:
        print(f"Error: {CONFIG_FILE} cannot be imported even after creation attempt.")
        return True, [] # Default to defining ROIs if load fails critically
    except AttributeError as e:
         print(f"Error accessing attributes in {CONFIG_FILE}. Ensure DEFINE_ROI and ROI_ZONES exist.")
         print(f"Details: {e}")
         print("Assuming default config values (DEFINE_ROI=True, ROI_ZONES=[]).")
         return True, []
    except Exception as e:
        print(f"Error loading config: {e}")
        # Decide whether to exit or return defaults
        print("Returning default config values due to error.")
        return True, []


def create_default_config():
    """Creates a default config.py file if it doesn't exist."""
    if not os.path.exists(CONFIG_FILE):
        print(f"Creating default {CONFIG_FILE}...")
        try:
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
        except IOError as e:
            print(f"Error: Could not create {CONFIG_FILE}. Please check permissions.")
            print(f"Details: {e}")
            # Depending on severity, you might want sys.exit(1) here


def save_config(roi_zones_to_save):
    """Saves the defined ROIs to config.py and sets DEFINE_ROI to False."""
    print("Saving ROIs to config.py...")
    try:
        with open(CONFIG_FILE, 'w') as f:
            f.write("# config.py\n\n")
            f.write("# Set to True to run the ROI definition process on the next run.\n")
            f.write("# Set to False to use the existing ROI_ZONES and skip definition.\n")
            f.write("DEFINE_ROI = False # Set to False after defining ROIs\n\n")
            f.write("# List to store the coordinates of the polygonal ROI zones.\n")
            f.write("# Each zone is a list of (x, y) tuples.\n")
            formatted_zones = repr(roi_zones_to_save) # Use repr for safe representation
            f.write(f"ROI_ZONES = {formatted_zones}\n")
        print("Configuration saved.")
    except IOError as e:
         print(f"Error saving config to {CONFIG_FILE}: {e}. Check permissions.")
    except Exception as e:
        print(f"Error saving config: {e}")

# --- ROI Definition Functions ---

def _draw_polygon_callback(event, x, y, flags, param):
    """Internal mouse callback for the ROI definition window."""
    # Access globals defined within this module
    global current_points, polygons, frame_copy_for_drawing

    if frame_copy_for_drawing is None:
        return

    display_frame = frame_copy_for_drawing.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point and provide visual feedback
        current_points.append((x, y))
        # print(f"Added point: ({x}, {y}). Current points: {len(current_points)}") # Optional debug print
        if len(current_points) > 0:
             for point in current_points:
                  cv2.circle(display_frame, point, 5, TEMP_LINE_COLOR, -1)
        if len(current_points) > 1:
            cv2.polylines(display_frame, [np.array(current_points)], isClosed=False, color=TEMP_LINE_COLOR, thickness=THICKNESS)
        cv2.imshow(WINDOW_NAME_DEFINE, display_frame)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Finalize current polygon if valid
        if len(current_points) >= 3:
            polygons.append(list(current_points)) # Store as list of tuples
            print(f"Polygon {len(polygons)} completed with {len(current_points)} points.")
            current_points = [] # Reset for next polygon

            # Redraw frame with all completed polygons shown clearly
            frame_copy_updated = frame_copy_for_drawing.copy()
            for poly in polygons:
                 cv2.polylines(frame_copy_updated, [np.array(poly)], isClosed=True, color=ROI_DRAW_COLOR, thickness=THICKNESS)
            cv2.imshow(WINDOW_NAME_DEFINE, frame_copy_updated)
        else:
            print("Need at least 3 points to define a polygon. Right-click ignored.")

def define_rois(capture_source_path):
    """
    Opens the first frame of a video source and allows the user to interactively
    define multiple polygonal ROIs. Saves defined ROIs to config.py.

    Args:
        capture_source_path (str or int): Path to video file or webcam index.

    Returns:
        list or None: A list of defined polygons (each a list of (x,y) tuples)
                      if saved successfully, otherwise None if cancelled or failed.
    """
    # Use module globals to track state during definition
    global current_points, polygons, frame_copy_for_drawing

    print(f"\n--- Starting ROI Definition for: {capture_source_path} ---")
    cap_roi = cv2.VideoCapture(capture_source_path)
    if not cap_roi.isOpened():
        print(f"Error: Could not open video source for ROI definition: {capture_source_path}")
        return None

    ret, frame = cap_roi.read()
    if not ret:
        print("Error: Could not read frame from video source for ROI definition.")
        cap_roi.release()
        return None

    # Release capture object - only needed the first frame
    cap_roi.release()

    # Reset state for this definition session
    frame_copy_for_drawing = frame.copy()
    polygons = []
    current_points = []

    cv2.namedWindow(WINDOW_NAME_DEFINE)
    cv2.setMouseCallback(WINDOW_NAME_DEFINE, _draw_polygon_callback) # Use internal callback

    print("INSTRUCTIONS:")
    print("  Left Mouse Click : Add point to current polygon.")
    print("  Right Mouse Click: Finish current polygon (needs >= 3 points).")
    print("  Key 's'        : Save all defined polygons and exit definition.")
    print("  Key 'r'        : Reset points for the polygon currently being drawn.")
    print("  Key 'c'        : Clear ALL defined polygons and start over.")
    print("  Key 'q'        : Quit definition mode WITHOUT saving.")
    print("----------------------------------------------------------")

    while True:
        # Prepare the display frame for this iteration
        display_frame = frame_copy_for_drawing.copy()

        # Draw already completed polygons
        if polygons:
            for poly in polygons:
                 cv2.polylines(display_frame, [np.array(poly)], isClosed=True, color=ROI_DRAW_COLOR, thickness=THICKNESS)

        # Draw the current polygon being built (points and connecting lines)
        if len(current_points) > 0:
            for point in current_points:
                cv2.circle(display_frame, point, 5, TEMP_LINE_COLOR, -1)
            if len(current_points) > 1:
                cv2.polylines(display_frame, [np.array(current_points)], isClosed=False, color=TEMP_LINE_COLOR, thickness=THICKNESS)

        # Show the frame with current state
        cv2.imshow(WINDOW_NAME_DEFINE, display_frame)
        key = cv2.waitKey(20) & 0xFF # 20ms wait allows UI responsiveness

        # Handle key presses for actions
        if key == ord('s'):
            if not polygons:
                print("No polygons defined to save.")
            else:
                save_config(polygons) # Call save function
                break # Exit definition loop successfully

        elif key == ord('r'):
            print("Resetting points for the current polygon.")
            current_points = []
            # No need to redraw explicitly here, the next loop iteration will redraw correctly

        elif key == ord('c'):
            print("Clearing all defined polygons.")
            polygons = []
            current_points = []
            # frame_copy_for_drawing remains the original clean frame

        elif key == ord('q'):
            print("Quitting ROI definition without saving.")
            polygons = None # Indicate cancellation explicitly
            break # Exit definition loop

    # Cleanup
    cv2.destroyWindow(WINDOW_NAME_DEFINE)
    frame_copy_for_drawing = None # Clear frame copy

    return polygons # Return the list of polygons or None