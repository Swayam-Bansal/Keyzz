import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

# --- (Keep the rest of your initializations the same: pygame, NOTES, sounds, MediaPipe, globals) ---
# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Define piano notes (C4 to B4)
NOTES = {
    0: 'C4', 
    # 1: 'D4', 2: 'E4', 3: 'F4', 4: 'G4', 5: 'A4', 6: 'B4',
    # 7: 'C#4', 8: 'D#4', 9: 'F#4', 10: 'G#4', 11: 'A#4'
}

# Mapping from black key visual index (0-4) to Note Index (7-11)
black_key_note_indices = [7, 8, 9, 10, 11] # C#, D#, F#, G#, A#
black_key_visual_to_note = {
    visual_idx: note_idx
    for visual_idx, note_idx in enumerate(black_key_note_indices)
}

# Dot Detection ROI Ratios (relative to key/image dimensions)
# For black keys (at bottom): search within the black key vertical space
black_key_dot_roi_y_start_ratio = 0.1 # Start searching 10% down from the top *of the black key*
black_key_dot_roi_y_end_ratio = 0.9   # Stop searching 10% up from the bottom *of the black key*
# For white keys (at top): search near the top edge
white_key_dot_roi_y_start_ratio = 0.05 # Start searching 5% down from the top *of the image*
white_key_dot_roi_y_end_ratio = 0.4   # Stop searching 40% down from the top *of the image*
# -----------------------------


# Load sound files (placeholder for actual sound files)
sounds = {}
print("Loading/Generating sounds...")
for key, note in NOTES.items():
    try:
        sounds[key] = pygame.mixer.Sound(f"sounds/{note}.wav")
        print(f"Loaded sound for {note}")
    except pygame.error: # Catch pygame specific error for file not found
        print(f"Warning: Sound file for {note} not found. Creating placeholder.")
        # Create a simple sine wave for each note if file not found
        sample_rate = 44100
        duration = 0.5  # seconds
        # Calculate frequency using C4=261.63 and equal temperament
        frequency = 261.63 * (2 ** ((key if key < 7 else key - 7 + (0 if key==7 else 1 if key==8 else 3 if key==9 else 4 if key==10 else 5)) / 12.0))
        # Adjust frequency for black keys correctly relative to C4
        # C#: key=7 -> 1/12 | D#: key=8 -> 3/12 | F#: key=9 -> 6/12 | G#: key=10 -> 8/12 | A#: key=11 -> 10/12 steps from C4
        if key == 7: frequency = 261.63 * (2**(1/12.0)) # C#4
        elif key == 8: frequency = 261.63 * (2**(3/12.0)) # D#4
        elif key == 9: frequency = 261.63 * (2**(6/12.0)) # F#4
        elif key == 10: frequency = 261.63 * (2**(8/12.0)) # G#4
        elif key == 11: frequency = 261.63 * (2**(10/12.0)) # A#4
        elif key > 0: frequency = 261.63 * (2**(key * 2 / 12.0 if key <= 2 else (key*2-1)/12.0)) # D, E, F, G, A, B

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t) # Reduced amplitude slightly
        # Ensure stereo, int16 format
        audio = np.asarray([32767 * sine_wave, 32767 * sine_wave]).T.astype(np.int16)
        sounds[key] = pygame.sndarray.make_sound(audio)
print("Sound loading complete.")


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global variables for stabilized piano detection
stabilized_sheet = None
stabilized_corners = None
stabilized_perspective_transform = None
stabilized_white_keys = None
stabilized_black_keys = None
stabilized_white_dots = None
stabilized_black_dots = None
stabilization_counter = 0
MAX_STABILIZATION_FRAMES = 30
is_stabilized = False
canvas_width = 700  # Target width for perspective-corrected view
canvas_height = 200  # Target height for perspective-corrected view

# --- (Keep detect_piano_sheet_corners, apply_perspective_correction, detect_keys_from_sheet, detect_dots the same) ---
def detect_piano_sheet_corners(image):
    """Detect the four corners of the piano sheet for perspective transformation"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area - looking for the piano sheet
    if contours:
        # Sort contours by area (largest first) and take the top 5
        candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in candidates:
            # Check if contour is approximately rectangular (piano sheet likely is)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # If we have a quadrilateral
                # Sort corners in the correct order (top-left, top-right, bottom-right, bottom-left)
                corners = np.array([point[0] for point in approx])

                # Check if it has a reasonable aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

                if 1.5 < aspect_ratio < 6.0:  # Adjusted aspect ratio slightly
                    # Calculate center of each point
                    sums = corners.sum(axis=1)
                    diffs = np.diff(corners, axis=1) # Use np.diff for difference calculation

                    # Sort corners: top-left, top-right, bottom-right, bottom-left
                    ordered_corners = np.zeros((4, 2), dtype=np.int32) # Use int32
                    ordered_corners[0] = corners[np.argmin(sums)]  # Top-left has smallest sum
                    ordered_corners[2] = corners[np.argmax(sums)]  # Bottom-right has largest sum
                    ordered_corners[1] = corners[np.argmin(diffs)]  # Top-right has smallest difference (x-y)
                    ordered_corners[3] = corners[np.argmax(diffs)]  # Bottom-left has largest difference (x-y)

                    # Ensure the order is correct based on position (more robust)
                    # Sort by y-coordinate first, then x-coordinate
                    corners_sorted_y = corners[np.argsort(corners[:, 1])]
                    top_corners = corners_sorted_y[:2]
                    bottom_corners = corners_sorted_y[2:]

                    # Sort top corners by x-coordinate
                    top_corners_sorted_x = top_corners[np.argsort(top_corners[:, 0])]
                    tl = top_corners_sorted_x[0]
                    tr = top_corners_sorted_x[1]

                    # Sort bottom corners by x-coordinate
                    bottom_corners_sorted_x = bottom_corners[np.argsort(bottom_corners[:, 0])]
                    bl = bottom_corners_sorted_x[0]
                    br = bottom_corners_sorted_x[1]

                    ordered_corners_robust = np.array([tl, tr, br, bl], dtype=np.int32)


                    return ordered_corners_robust, cv2.boundingRect(contour) # Return robust ordering

        # Fallback if no perfect rectangle found
        if candidates and len(candidates[0]) >= 4:
            peri = cv2.arcLength(candidates[0], True)
            approx = cv2.approxPolyDP(candidates[0], 0.02 * peri, True)
            # If approxPolyDP gives 4 points, use that
            if len(approx) == 4:
                 corners = np.array([point[0] for point in approx], dtype=np.int32)
                 # Attempt to sort these 4 points
                 sums = corners.sum(axis=1)
                 diffs = np.diff(corners, axis=1)
                 ordered_corners = np.zeros((4, 2), dtype=np.int32)
                 ordered_corners[0] = corners[np.argmin(sums)]
                 ordered_corners[2] = corners[np.argmax(sums)]
                 ordered_corners[1] = corners[np.argmin(diffs)]
                 ordered_corners[3] = corners[np.argmax(diffs)]
                 # Add robust sorting here as well if needed
                 return ordered_corners, cv2.boundingRect(candidates[0])

            # If not 4 points from approxPolyDP, try convex hull method (less reliable)
            if len(approx) >= 4:
                hull = cv2.convexHull(approx)
                if len(hull) >= 4:
                    # Simple bounding box corners as fallback if hull isn't 4 points
                    rect = cv2.boundingRect(candidates[0])
                    x, y, w, h = rect
                    corners = np.array([[x,y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                    return corners, rect

    return None, None

def apply_perspective_correction(image, corners):
    """Apply perspective correction to get a bird's eye view of the piano"""
    global canvas_width, canvas_height

    if corners is None or len(corners) != 4: # Ensure we have 4 corners
        return None, None

    # Define the corners of the destination image
    dst_corners = np.array([
        [0, 0],  # Top-left
        [canvas_width, 0],  # Top-right
        [canvas_width, canvas_height],  # Bottom-right
        [0, canvas_height]  # Bottom-left
    ], dtype=np.float32)

    # Convert corners to the required format
    src_corners = corners.astype(np.float32) # Ensure float32

    # Calculate the perspective transform matrix
    try:
        perspective_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    except cv2.error as e:
        print(f"Error getting perspective transform: {e}")
        print(f"Source Corners:\n{src_corners}")
        return None, None # Return None if matrix calculation fails

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, perspective_matrix, (canvas_width, canvas_height))

    return warped, perspective_matrix

# (Keep imports, initializations, other functions like detect_piano_sheet_corners,
#  apply_perspective_correction, transform_point, stabilize_piano_sheet, etc. the same
#  as the previous corrected version)

def detect_keys_from_sheet(warped_image):
    """
    Detect piano keys from the perspective-corrected piano sheet,
    assuming an inverted view (black keys at the bottom).
    """
    if warped_image is None:
        return [], []

    # Get dimensions of the warped image
    h, w = warped_image.shape[:2]
    if h == 0 or w == 0: return [], []

    # Define the number of white keys
    num_white_keys = 7

    # --- White Keys ---
    # White keys occupy the full visual space vertically underneath black keys
    white_keys = []
    white_key_width = w / num_white_keys  # Use float division for accuracy

    for i in range(num_white_keys):
        key_x = int(i * white_key_width)
        next_key_x = int((i + 1) * white_key_width)
        current_width = next_key_x - key_x  # Calculate width accurately
        # Define white key area covering the full height for touch detection
        white_keys.append((key_x, 0, current_width, h))

    # --- Black Keys ---
    # Black keys are positioned at the BOTTOM of the warped image
    black_keys = []
    black_key_height_ratio = 0.6  # Black keys occupy bottom 60% of height
    black_key_height = int(h * black_key_height_ratio)
    black_key_width = int(white_key_width * 0.6)
    # Y-coordinate where black keys START (from the top)
    black_key_start_y = h - black_key_height

    # Positions relative to the START of the white key index they are 'after'
    # C# (after 0), D# (after 1), F# (after 3), G# (after 4), A# (after 5)
    black_key_positions_indices = [0, 1, 3, 4, 5] # White key index before the black key

    for idx in black_key_positions_indices:
        if idx < num_white_keys - 1: # Ensure we don't go past the last white key boundary
            # Calculate center based on the boundary between white keys idx and idx+1
            boundary_x = (idx + 1) * white_key_width
            black_cx = int(boundary_x) # Center black key on the line between white keys
            black_x = int(black_cx - black_key_width / 2)
            # Position black keys starting from the bottom part of the image
            black_keys.append((black_x, black_key_start_y, black_key_width, black_key_height))

    return white_keys, black_keys

def detect_dots(warped_image, white_keys, black_keys):
    """
    Detect the white and black dots on the corrected piano sheet,
    assuming an inverted view (dots on white keys near top, dots on black keys within bottom area).
    """
    if warped_image is None:
        return [], []

    gray_roi = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    h, w = gray_roi.shape # Get height for ROI calculation

    # Lists to store dots with their key indices
    white_dots = []  # White dots (expected on black keys - bottom area)
    black_dots = []  # Black dots (expected on white keys - top area)

    # --- Dot Detection Parameters ---
    min_dot_area = 15
    max_dot_area = 250
    white_dot_threshold = 180 # Threshold for finding light areas (dots)
    black_dot_threshold = 70  # Threshold for finding dark areas (dots)

    # ROI ratios - adjust based on where dots are placed on your sheet
    # For black keys (at bottom): search within the black key vertical space
    # black_key_dot_roi_y_start_ratio = 0.1 # Start searching 10% down from the top *of the black key*
    # black_key_dot_roi_y_end_ratio = 0.9   # Stop searching 10% up from the bottom *of the black key*
    # # For white keys (at top): search near the top edge
    # white_key_dot_roi_y_start_ratio = 0.05 # Start searching 5% down from the top *of the image*
    # white_key_dot_roi_y_end_ratio = 0.4   # Stop searching 40% down from the top *of the image*

    # Mapping black key visual index to NOTES index
    # black_key_note_indices = [7, 8, 9, 10, 11] # C#, D#, F#, G#, A#
    # black_key_visual_to_note = {
    #     visual_idx: note_idx
    #     for visual_idx, note_idx in enumerate(black_key_note_indices)
    # }

    # Process black keys (bottom area) to find white dots
    for i, (bx, by, bw, bh) in enumerate(black_keys):
        # Define ROI within this specific black key's bounding box
        roi_y_start = int(by + bh * black_key_dot_roi_y_start_ratio)
        roi_y_end = int(by + bh * black_key_dot_roi_y_end_ratio)
        roi_x_start = bx
        roi_x_end = bx + bw

        # Ensure ROI coordinates are valid
        roi_y_start = max(0, roi_y_start)
        roi_x_start = max(0, roi_x_start)
        roi_y_end = min(h, roi_y_end)
        roi_x_end = min(w, roi_x_end)

        if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start: continue # Skip if ROI is invalid

        key_roi = gray_roi[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        if key_roi.size == 0: continue

        # Threshold to find white things in the ROI
        _, white_thresh = cv2.threshold(key_roi, white_dot_threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in white_contours:
            area = cv2.contourArea(contour)
            if min_dot_area < area < max_dot_area:
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    # Calculate center relative to the ROI, then convert to full warped image coords
                    cx_roi = int(m["m10"] / m["m00"])
                    cy_roi = int(m["m01"] / m["m00"])
                    cx = cx_roi + roi_x_start
                    cy = cy_roi + roi_y_start

                    # Get the correct note index for this black key visual index 'i'
                    note_idx = black_key_visual_to_note.get(i)
                    if note_idx is not None:
                        white_dots.append((cx, cy, note_idx))

    # Process white keys (top area) to find black dots
    for i, (wx, wy, ww, wh) in enumerate(white_keys):
        # Define ROI within the top portion of the white key's area
        roi_y_start = int(wy + wh * white_key_dot_roi_y_start_ratio) # wy is 0
        roi_y_end = int(wy + wh * white_key_dot_roi_y_end_ratio)   # wh is image height
        roi_x_start = wx
        roi_x_end = wx + ww

        # Ensure ROI coordinates are valid
        roi_y_start = max(0, roi_y_start)
        roi_x_start = max(0, roi_x_start)
        roi_y_end = min(h, roi_y_end) # Clamp to image height
        roi_x_end = min(w, roi_x_end)

        if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start: continue # Skip if ROI is invalid

        key_roi = gray_roi[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        if key_roi.size == 0: continue

        # Threshold to find dark things in the ROI
        _, black_thresh = cv2.threshold(key_roi, black_dot_threshold, 255, cv2.THRESH_BINARY_INV) # Inverted threshold

        # Find contours
        black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in black_contours:
            area = cv2.contourArea(contour)
            if min_dot_area < area < max_dot_area:
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    # Calculate center relative to ROI, then convert to full warped image coords
                    cx_roi = int(m["m10"] / m["m00"])
                    cy_roi = int(m["m01"] / m["m00"])
                    cx = cx_roi + roi_x_start
                    cy = cy_roi + roi_y_start
                    # White key index 'i' directly corresponds to note index 0-6
                    black_dots.append((cx, cy, i))

    return white_dots, black_dots


def is_key_pressed(finger_tip, dots, threshold=25): # Reduced threshold slightly
    """Check if a finger is pressing near a dot"""
    if finger_tip is None: return False, None # Handle None input

    closest_dot_dist = float('inf')
    pressed_key_idx = None

    for dot_x, dot_y, key_idx in dots:
        distance = np.sqrt((finger_tip[0] - dot_x)**2 + (finger_tip[1] - dot_y)**2)
        if distance < threshold and distance < closest_dot_dist:
             closest_dot_dist = distance
             pressed_key_idx = key_idx

    if pressed_key_idx is not None:
        return True, pressed_key_idx
    else:
        return False, None

# CORRECTED transform_point function
def transform_point(point, perspective_matrix):
    """Transform a point using the perspective transformation matrix"""
    if perspective_matrix is None:
        # print("Warning: Attempted to transform point with None matrix.")
        return None # Cannot transform without a matrix

    # Create point array in the correct format for cv2.perspectiveTransform
    # Shape: (1, 1, 2), dtype: float32
    src_point = np.array([[(point[0], point[1])]], dtype=np.float32)

    # Apply the transformation
    transformed = cv2.perspectiveTransform(src_point, perspective_matrix)

    # Check if the transformation was successful and output is valid
    if transformed is None or transformed.size == 0 or transformed.shape != (1, 1, 2):
        # print(f"Warning: Perspective transformation failed or returned invalid shape for point {point}. Output: {transformed}")
        return None # Indicate failure

    # Extract the transformed point (it will be inside nested arrays)
    warped_x = int(transformed[0][0][0])
    warped_y = int(transformed[0][0][1])

    return (warped_x, warped_y)

# CORRECTED inverse_transform_point function
def inverse_transform_point(point, perspective_matrix):
    """Transform a point from warped space back to original image space"""
    if perspective_matrix is None:
        # print("Warning: Attempted inverse transform with None matrix.")
        return None

    # Get inverse transformation matrix
    try:
        # Use cv2.invert for potentially better handling of matrix inversion
        retval, inverse_matrix = cv2.invert(perspective_matrix)
        if not retval:
             print("Error: Could not invert perspective matrix (cv2.invert failed).")
             return None
    except cv2.error as e:
        print(f"Error inverting perspective matrix: {e}")
        return None
    except np.linalg.LinAlgError: # Keep this as a fallback
        print("Error: Could not invert perspective matrix (LinAlgError).")
        return None

    # Create point array in the correct format
    # Shape: (1, 1, 2), dtype: float32
    src_point = np.array([[(point[0], point[1])]], dtype=np.float32)

    # Apply the inverse transformation
    transformed = cv2.perspectiveTransform(src_point, inverse_matrix)

    # Check if the transformation was successful and output is valid
    if transformed is None or transformed.size == 0 or transformed.shape != (1, 1, 2):
        # print(f"Warning: Inverse perspective transformation failed or returned invalid shape for point {point}. Output: {transformed}")
        return None # Indicate failure

    # Extract the transformed point
    orig_x = int(transformed[0][0][0])
    orig_y = int(transformed[0][0][1])

    return (orig_x, orig_y)

# --- (Keep stabilize_piano_sheet the same) ---
def stabilize_piano_sheet(image, current_corners, current_rect):
    """Attempt to stabilize the piano sheet detection"""
    global stabilized_corners, stabilized_perspective_transform
    global stabilization_counter, is_stabilized, stabilized_sheet # Added stabilized_sheet

    # If we don't have a current detection, reset stabilization
    if current_corners is None or current_rect is None:
        stabilization_counter = 0
        is_stabilized = False
        # Reset stabilized values as well if detection lost
        stabilized_corners = None
        stabilized_sheet = None
        stabilized_perspective_transform = None
        print("Detection lost, resetting stabilization.")
        return None, None, None, None

    # Apply perspective correction early to get the matrix for comparison/return
    warped, current_transform_matrix = apply_perspective_correction(image, current_corners)
    if warped is None or current_transform_matrix is None:
        # If correction fails for current frame, can't stabilize based on it
        stabilization_counter = 0
        is_stabilized = False
        print("Perspective correction failed for current frame, resetting stabilization.")
        return None, None, None, None

    # If we're already stabilized, just return the stable values and the *newly* warped image
    if is_stabilized:
        # Re-warp the *current* frame using the *stabilized* transform
        stable_warped, _ = apply_perspective_correction(image, stabilized_corners)
        if stable_warped is None:
            print("Warning: Failed to warp current frame using stabilized transform.")
            # Optional: could try to reset stabilization here
        return stabilized_corners, stabilized_sheet, stabilized_perspective_transform, stable_warped

    # If this is our first good detection, initialize stabilization
    if stabilization_counter == 0:
        stabilized_corners = current_corners.copy() # Use copy
        stabilized_sheet = current_rect
        stabilized_perspective_transform = current_transform_matrix.copy() # Use copy
        stabilization_counter += 1
        print(f"Stabilization started (1/{MAX_STABILIZATION_FRAMES})")
        return stabilized_corners, stabilized_sheet, stabilized_perspective_transform, warped

    # --- Stabilization Logic ---
    # Check if current detection is similar to stabilized values

    # 1. Compare Corner Positions
    corner_distances = np.sqrt(np.sum((current_corners - stabilized_corners)**2, axis=1))
    avg_corner_dist = np.mean(corner_distances)

    # 2. Compare Rectangle Centers and Area (as before)
    x1, y1, w1, h1 = current_rect
    x2, y2, w2, h2 = stabilized_sheet
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    area1 = w1*h1
    area2 = w2*h2
    area_ratio = area1 / area2 if area2 > 0 else 0

    # Define thresholds for stability
    MAX_AVG_CORNER_DIST = 20 # Max average pixels corners can move
    MAX_CENTER_DIST = 30     # Max pixels center can move
    MIN_AREA_RATIO = 0.8
    MAX_AREA_RATIO = 1.2

    # If the detection is consistent
    if (avg_corner_dist < MAX_AVG_CORNER_DIST and
        center_distance < MAX_CENTER_DIST and
        MIN_AREA_RATIO < area_ratio < MAX_AREA_RATIO):

        stabilization_counter += 1

        # Smooth the corners (Exponential Moving Average)
        alpha = 0.1 # Smoothing factor (lower means more smoothing)
        smoothed_corners = alpha * current_corners + (1 - alpha) * stabilized_corners
        stabilized_corners = smoothed_corners.astype(np.int32)

        # Update the stabilized rectangle (maybe average?)
        stabilized_sheet = current_rect # Or average x,y,w,h if desired

        # Recalculate perspective transform with smoothed corners
        smoothed_warped, smoothed_transform_matrix = apply_perspective_correction(image, stabilized_corners)

        if smoothed_warped is None or smoothed_transform_matrix is None:
             print("Warning: Perspective correction failed with smoothed corners.")
             # Don't update transform if smoothing caused failure, maybe reset?
             stabilization_counter = 0 # Reset if smoothing breaks it
             is_stabilized = False
             return current_corners, current_rect, current_transform_matrix, warped # Return current state
        else:
             stabilized_perspective_transform = smoothed_transform_matrix.copy() # Update stable transform
             warped = smoothed_warped # Use the smoothed warp for this frame


        print(f"Stabilizing... ({stabilization_counter}/{MAX_STABILIZATION_FRAMES})")

        # If we've reached the threshold, mark as stabilized
        if stabilization_counter >= MAX_STABILIZATION_FRAMES:
            is_stabilized = True
            print("Piano keyboard detection stabilized!")

        return stabilized_corners, stabilized_sheet, stabilized_perspective_transform, warped # Return stabilized state

    else:
        # If detection is significantly different, reset stabilization
        print(f"Detection unstable (Avg Corner Dist: {avg_corner_dist:.1f}, Center Dist: {center_distance:.1f}, Area Ratio: {area_ratio:.2f}). Resetting stabilization.")
        stabilization_counter = 0
        is_stabilized = False
        # Re-initialize with the current (unstable) detection as the new base
        stabilized_corners = current_corners.copy()
        stabilized_sheet = current_rect
        stabilized_perspective_transform = current_transform_matrix.copy()
        # Return the current (non-stabilized) state for this frame
        return current_corners, current_rect, current_transform_matrix, warped


# --- Main Loop (with modifications for error handling) ---
def main():
    global is_stabilized, stabilization_counter # Allow main to modify these
    global stabilized_corners, stabilized_sheet, stabilized_perspective_transform # Needed if reset happens
    global stabilized_white_keys, stabilized_black_keys # Cache these when stabilized
    global stabilized_white_dots, stabilized_black_dots

    cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work

    if not cap.isOpened():
       print("Error: Could not open video source.")
       return

    previously_pressed_keys = set()
    last_press_time = {}
    debounce_interval = 0.2 # Seconds between same key presses

    print("Waiting for camera...")
    time.sleep(2.0) # Give camera more time

    # --- Create Windows ---
    cv2.namedWindow('Piano Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Corrected Piano View', cv2.WINDOW_NORMAL)


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            time.sleep(0.1) # Wait a bit before retrying
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        h, w, _ = image.shape

        # --- Local variables for this frame ---
        current_corners = None
        current_rect = None
        current_transform_matrix = None
        current_warped_piano = None
        vis_piano = None # Visualization window content

        # --- Piano Detection and Stabilization ---
        detected_corners, detected_rect = detect_piano_sheet_corners(image)

        if detected_corners is not None and detected_rect is not None:
             # Attempt to stabilize or use stabilized values
             stable_corners, stable_rect, stable_transform, stable_warped = stabilize_piano_sheet(image, detected_corners, detected_rect)

             # Use the results from stabilization
             current_corners = stable_corners
             current_rect = stable_rect
             current_transform_matrix = stable_transform
             current_warped_piano = stable_warped

        elif is_stabilized:
             # If detection lost BUT we WERE stabilized, keep using old transform but reset state
             print("Detection lost, resetting stabilization state.")
             is_stabilized = False
             stabilization_counter = 0
             # Keep the last known good transform for this frame? Or None? Let's use None.
             current_transform_matrix = None # Indicate no valid transform this frame
             current_warped_piano = None
             # Reset cached keys/dots
             stabilized_white_keys = None
             stabilized_black_keys = None
             stabilized_white_dots = None
             stabilized_black_dots = None
        else:
             # Not stabilized and no detection this frame
             current_transform_matrix = None
             current_warped_piano = None


        # --- Key and Dot Detection (only if we have a valid warped image) ---
        white_keys, black_keys = [], []
        white_dots, black_dots = [], []

        if current_warped_piano is not None:
            if is_stabilized and stabilized_white_keys is not None:
                # Use cached keys/dots if stabilized and previously calculated
                white_keys = stabilized_white_keys
                black_keys = stabilized_black_keys
                white_dots = stabilized_white_dots
                black_dots = stabilized_black_dots
            else:
                # Detect keys and dots from the current warped image
                white_keys, black_keys = detect_keys_from_sheet(current_warped_piano)
                white_dots, black_dots = detect_dots(current_warped_piano, white_keys, black_keys)
                # If now stabilized, cache these results
                if is_stabilized:
                    stabilized_white_keys = white_keys
                    stabilized_black_keys = black_keys
                    stabilized_white_dots = white_dots
                    stabilized_black_dots = black_dots

            # --- Prepare Visualization Window ---
            vis_piano = current_warped_piano.copy()
            # Draw white keys visually (e.g., as background rectangles)
            for wx, wy, ww, wh in white_keys:
                # Draw full white key area subtly for reference
                cv2.rectangle(vis_piano, (wx, wy), (wx + ww, wy + wh), (220, 220, 220), -1) # Light gray fill
                cv2.rectangle(vis_piano, (wx, wy), (wx + ww, wy + wh), (180, 180, 180), 1) # Slightly darker outline

            # Draw black keys (at the bottom) on top
            for bx, by, bw, bh in black_keys:
                cv2.rectangle(vis_piano, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1) # Filled black
                cv2.rectangle(vis_piano, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1) # Outline

            # Draw dots on vis_piano (dot drawing logic remains the same)
            dot_radius = 6
            # White dots (on black keys area)
            for dx, dy, key_idx in white_dots:
                cv2.circle(vis_piano, (dx, dy), dot_radius, (255, 255, 255), -1)
                cv2.circle(vis_piano, (dx, dy), dot_radius, (0, 0, 0), 1)
            # Black dots (on white keys area)
            for dx, dy, key_idx in black_dots:
                cv2.circle(vis_piano, (dx, dy), dot_radius, (0, 0, 0), -1)
                cv2.circle(vis_piano, (dx, dy), dot_radius, (255, 255, 255), 1)


        elif not is_stabilized:
             # Create a blank canvas if no warped image and not stabilized
             vis_piano = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
             cv2.putText(vis_piano, "No Piano Detected", (canvas_width//2 - 100, canvas_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)


        # --- Hand Landmark Processing and Key Press Detection ---
        currently_pressed_keys = set()
        if results.multi_hand_landmarks and current_transform_matrix is not None and vis_piano is not None:
            all_dots = white_dots + black_dots # Combine dot lists for checking

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on main image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2), # Smaller landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

                # Process finger tips
                finger_tips_landmarks = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
                    # Optional: Add THUMB_TIP if needed, might be less accurate for key pressing
                    # hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                ]

                for landmark in finger_tips_landmarks:
                    # Get original image coordinates
                    orig_x, orig_y = int(landmark.x * w), int(landmark.y * h)

                    # Transform to warped coordinates
                    transformed_coords = transform_point((orig_x, orig_y), current_transform_matrix)

                    if transformed_coords is not None:
                        warped_x, warped_y = transformed_coords

                        # Draw finger tip on both views
                        cv2.circle(image, (orig_x, orig_y), 5, (255, 255, 0), -1) # Cyan on main view
                        if 0 <= warped_x < canvas_width and 0 <= warped_y < canvas_height:
                            cv2.circle(vis_piano, (warped_x, warped_y), 5, (255, 255, 0), -1) # Cyan on warped view

                        # Check for key press near this finger tip
                        is_pressed, key_idx = is_key_pressed((warped_x, warped_y), all_dots)
                        if is_pressed:
                            currently_pressed_keys.add(key_idx)
                            # Highlight pressed key on vis_piano
                            if key_idx < 7 and key_idx < len(white_keys): # White key
                                wx, wy, ww, wh = white_keys[key_idx]
                                # Highlight the *top* part of the white key visually
                                highlight_y_end = int(h * white_key_dot_roi_y_end_ratio) # Match dot search area
                                cv2.rectangle(vis_piano, (wx, wy), (wx + ww, highlight_y_end), (0, 255, 255), 2) # Yellow highlight in top area
                            elif key_idx >= 7: # Black key
                                # Find the corresponding black key visual index
                                visual_idx = -1
                                for v_idx, n_idx in black_key_visual_to_note.items():
                                    if n_idx == key_idx:
                                        visual_idx = v_idx
                                        break
                                if visual_idx != -1 and visual_idx < len(black_keys):
                                     bx, by, bw, bh = black_keys[visual_idx]
                                     # Highlight the black key itself
                                     cv2.rectangle(vis_piano, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2) # Yellow highlight



        # --- Play Sounds ---
        current_time = time.time()
        for key in currently_pressed_keys - previously_pressed_keys:
            if key in sounds and (current_time - last_press_time.get(key, 0)) > debounce_interval:
                sounds[key].play()
                last_press_time[key] = current_time
                print(f"Played: {NOTES.get(key, 'Unknown')}")

        previously_pressed_keys = currently_pressed_keys

        # --- Draw Status Text on Main Image ---
        if is_stabilized:
            status_text = "Status: Stabilized - Playing"
            status_color = (0, 255, 0) # Green
        elif stabilization_counter > 0:
            progress = min(100, int(stabilization_counter / MAX_STABILIZATION_FRAMES * 100))
            status_text = f"Status: Stabilizing... {progress}%"
            status_color = (0, 255, 255) # Yellow
        else:
            status_text = "Status: Searching for Piano Sheet..."
            status_color = (0, 0, 255) # Red

        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Draw the detected corners outline on the main image
        if current_corners is not None:
            cv2.polylines(image, [current_corners.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)


        # --- Display Frames ---
        cv2.imshow('Piano Detection', image)
        if vis_piano is not None:
            cv2.imshow('Corrected Piano View', vis_piano)

        # --- Exit Condition ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'): # Add a key to reset stabilization manually
             print("Manual reset triggered.")
             is_stabilized = False
             stabilization_counter = 0
             stabilized_corners = None
             stabilized_sheet = None
             stabilized_perspective_transform = None
             stabilized_white_keys = None
             stabilized_black_keys = None
             stabilized_white_dots = None
             stabilized_black_dots = None


    # --- Cleanup ---
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pygame.mixer.quit()
    print("Done.")

if __name__ == "__main__":
    main()