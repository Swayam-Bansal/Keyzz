import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Define piano notes (C4 to B4)
NOTES = {
    0: 'C4', 
    # 1: 'D4', 2: 'E4', 3: 'F4', 4: 'G4', 5: 'A4', 6: 'B4',
    # 7: 'C#4', 8: 'D#4', 9: 'F#4', 10: 'G#4', 11: 'A#4'
}

# Load sound files (placeholder for actual sound files)
sounds = {}
for key, note in NOTES.items():
    try:
        sounds[key] = pygame.mixer.Sound(f"sounds/{note}.wav")
    except:
        print(f"Warning: Sound file for {note} not found. Creating placeholder.")
        # Create a simple sine wave for each note if file not found
        sample_rate = 44100
        duration = 0.5  # seconds
        frequency = 261.63 * (2 ** (key / 12.0))  # C4 = 261.63 Hz, with equal temperament
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        sine_wave = np.sin(2 * np.pi * frequency * t)
        audio = np.asarray([32767 * sine_wave, 32767 * sine_wave]).T.astype(np.int16)
        sounds[key] = pygame.mixer.Sound(pygame.sndarray.make_sound(audio))

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
                
                if 1.5 < aspect_ratio < 5.0:  # Piano keyboard is usually wider than tall
                    # Calculate center of each point
                    sums = corners.sum(axis=1)
                    diffs = corners[:, 0] - corners[:, 1]
                    
                    # Sort corners: top-left, top-right, bottom-right, bottom-left
                    ordered_corners = np.zeros_like(corners)
                    ordered_corners[0] = corners[np.argmin(sums)]  # Top-left has smallest sum
                    ordered_corners[2] = corners[np.argmax(sums)]  # Bottom-right has largest sum
                    ordered_corners[1] = corners[np.argmax(diffs)]  # Top-right has largest difference
                    ordered_corners[3] = corners[np.argmin(diffs)]  # Bottom-left has smallest difference
                    
                    return ordered_corners, cv2.boundingRect(contour)
        
        # If no suitable rectangle found, try to get corners from the largest contour
        if candidates and len(candidates[0]) >= 4:
            peri = cv2.arcLength(candidates[0], True)
            approx = cv2.approxPolyDP(candidates[0], 0.02 * peri, True)
            if len(approx) >= 4:
                # Get the four most extreme points
                hull = cv2.convexHull(approx)
                leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
                rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
                topmost = tuple(hull[hull[:, :, 1].argmin()][0])
                bottommost = tuple(hull[hull[:, :, 1].argmax()][0])
                
                corners = np.array([topmost, rightmost, bottommost, leftmost])
                return corners, cv2.boundingRect(candidates[0])
    
    return None, None

def apply_perspective_correction(image, corners):
    """Apply perspective correction to get a bird's eye view of the piano"""
    global canvas_width, canvas_height
    
    if corners is None:
        return None, None
    
    # Define the corners of the destination image
    dst_corners = np.array([
        [0, 0],  # Top-left
        [canvas_width, 0],  # Top-right
        [canvas_width, canvas_height],  # Bottom-right
        [0, canvas_height]  # Bottom-left
    ], dtype=np.float32)
    
    # Convert corners to the required format
    src_corners = corners.astype(np.float32)
    
    # Calculate the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, perspective_matrix, (canvas_width, canvas_height))
    
    return warped, perspective_matrix

def detect_keys_from_sheet(warped_image):
    """Detect piano keys from the perspective-corrected piano sheet"""
    if warped_image is None:
        return [], []
    
    # Get dimensions of the warped image
    h, w = warped_image.shape[:2]
    
    # Define the number of white keys
    num_white_keys = 7
    
    # Create white keys
    white_keys = []
    white_key_width = w // num_white_keys
    
    for i in range(num_white_keys):
        key_x = i * white_key_width
        white_keys.append((key_x, 0, white_key_width, h))
    
    # Create black keys
    # Standard piano has black keys at positions 0, 1, 3, 4, 5 (when counting spaces between white keys)
    black_keys = []
    black_key_width = int(white_key_width * 0.6)
    black_key_height = int(h * 0.6)
    black_key_positions = [0, 1, 3, 4, 5]  # C#, D#, F#, G#, A#
    
    for pos in black_key_positions:
        if pos >= num_white_keys - 1:
            continue  # Skip if position would be out of bounds
        black_x = white_key_width // 2 + pos * white_key_width
        # Position black keys at BOTTOM of white keys 
        black_y = h - black_key_height  # Position at bottom of white keys
        black_keys.append((int(black_x - black_key_width/2), black_y, black_key_width, black_key_height))
    
    return white_keys, black_keys

def detect_dots(warped_image, white_keys, black_keys):
    """Detect the white and black dots on the corrected piano sheet"""
    if warped_image is None:
        return [], []
    
    gray_roi = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    # Lists to store dots with their key indices
    white_dots = []  # Dots on black keys (will be white dots)
    black_dots = []  # Dots on white keys (will be black dots)
    
    # Process black keys to find white dots
    for i, (bx, by, bw, bh) in enumerate(black_keys):
        # Extract ROI for this black key
        key_roi = gray_roi[by:by+bh, bx:bx+bw]
        
        if key_roi.size == 0:
            continue
            
        # Threshold to find white dots on black keys
        _, white_thresh = cv2.threshold(key_roi, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 200:  # Adjust threshold based on your dot size
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"]) + bx
                    cy = int(m["m01"] / m["m00"]) + by
                    # Store the dot with its corresponding black key index
                    white_dots.append((cx, cy, i + 7))  # Offset by 7 for black keys
    
    # Process white keys to find black dots
    for i, (wx, wy, ww, wh) in enumerate(white_keys):
        # For inverted layout, check upper half of white keys for dots
        key_roi = gray_roi[wy:wy+wh//2, wx:wx+ww]
        
        if key_roi.size == 0:
            continue
            
        # Threshold to find black dots on white keys
        _, black_thresh = cv2.threshold(key_roi, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in black_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 200:  # Adjust threshold based on your dot size
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"]) + wx
                    cy = int(m["m01"] / m["m00"]) + wy
                    # Store the dot with its corresponding white key index
                    black_dots.append((cx, cy, i))
    
    return white_dots, black_dots

def is_key_pressed(finger_tip, dots, threshold=30):
    """Check if a finger is pressing near a dot"""
    for dot_x, dot_y, key_idx in dots:
        distance = np.sqrt((finger_tip[0] - dot_x)**2 + (finger_tip[1] - dot_y)**2)
        if distance < threshold:
            return True, key_idx
    return False, None

def transform_point(point, perspective_matrix):
    """Transform a point using the perspective transformation matrix"""
    # Create homogeneous coordinates
    homogeneous_point = np.array([[point[0], point[1], 1]], dtype=np.float32)
    
    # Apply the transformation
    transformed = cv2.perspectiveTransform(homogeneous_point.reshape(-1, 1, 3), perspective_matrix)
    
    # Return the transformed point
    return (int(transformed[0][0][0]), int(transformed[0][0][1]))

def inverse_transform_point(point, perspective_matrix):
    """Transform a point from warped space back to original image space"""
    # Get inverse transformation matrix
    inverse_matrix = np.linalg.inv(perspective_matrix)
    
    # Create homogeneous coordinates
    homogeneous_point = np.array([[point[0], point[1], 1]], dtype=np.float32)
    
    # Apply the inverse transformation
    transformed = cv2.perspectiveTransform(homogeneous_point.reshape(-1, 1, 3), inverse_matrix)
    
    # Return the transformed point
    return (int(transformed[0][0][0]), int(transformed[0][0][1]))

def stabilize_piano_sheet(image, current_corners, current_rect):
    """Attempt to stabilize the piano sheet detection"""
    global stabilized_sheet, stabilized_corners, stabilized_perspective_transform
    global stabilization_counter, is_stabilized
    
    # If we don't have a current detection, reset stabilization
    if current_corners is None or current_rect is None:
        stabilization_counter = 0
        is_stabilized = False
        return None, None, None, None
    
    # If we're already stabilized, return the stable values
    if is_stabilized:
        return stabilized_corners, stabilized_sheet, stabilized_perspective_transform
    
    # If this is our first detection, initialize stabilization
    if stabilization_counter == 0:
        stabilized_corners = current_corners
        stabilized_sheet = current_rect
        # Calculate initial perspective transform
        warped, transform_matrix = apply_perspective_correction(image, current_corners)
        stabilized_perspective_transform = transform_matrix
        stabilization_counter += 1
        return current_corners, current_rect, transform_matrix, warped
    
    # Check if current detection is similar to stabilized values
    # Calculate center points of current and stabilized rectangles
    x1, y1, w1, h1 = current_rect
    x2, y2, w2, h2 = stabilized_sheet
    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)
    
    # Calculate distance between centers
    center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # If the centers are close and the rectangle sizes are similar, count towards stabilization
    if (center_distance < 50) and (0.8 < (w1*h1)/(w2*h2) < 1.25):
        stabilization_counter += 1
        
        # Smooth the corners (weighted average)
        weight = min(0.9, stabilization_counter / MAX_STABILIZATION_FRAMES)
        smoothed_corners = stabilized_corners * weight + current_corners * (1 - weight)
        stabilized_corners = smoothed_corners.astype(np.int32)
        
        # Update the stabilized rectangle
        stabilized_sheet = current_rect
        
        # Calculate perspective transform with smoothed corners
        warped, transform_matrix = apply_perspective_correction(image, stabilized_corners)
        stabilized_perspective_transform = transform_matrix
        
        # If we've reached the threshold, mark as stabilized
        if stabilization_counter >= MAX_STABILIZATION_FRAMES:
            is_stabilized = True
            print("Piano keyboard detection stabilized!")
            
        return stabilized_corners, stabilized_sheet, stabilized_perspective_transform, warped
    else:
        # If detection is very different, reset stabilization
        stabilization_counter = 0
        is_stabilized = False
        stabilized_corners = current_corners
        stabilized_sheet = current_rect
        warped, transform_matrix = apply_perspective_correction(image, current_corners)
        stabilized_perspective_transform = transform_matrix
        return current_corners, current_rect, transform_matrix, warped

def main():
    # For webcam input
    cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work
    
    # Variables to track pressed keys and debounce
    previously_pressed_keys = set()
    last_press_time = {}
    
    # Wait for camera to warm up
    for _ in range(5):
        cap.read()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        # Flip the image horizontally for a mirrored view
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image for hand landmarks
        results = hands.process(image_rgb)
        
        # Create visualizations
        warped_piano = None
        transform_matrix = None
        white_keys, black_keys = [], []
        white_dots, black_dots = [], []
        
        # Detect the piano sheet corners (but only if not stabilized)
        if not is_stabilized:
            corners, rect = detect_piano_sheet_corners(image)
            if corners is not None and rect is not None:
                # Try to stabilize the detection
                corners, rect, transform_matrix, warped_piano = stabilize_piano_sheet(image, corners, rect)
                
                if warped_piano is not None:
                    # Detect piano keys in the perspective-corrected image
                    white_keys, black_keys = detect_keys_from_sheet(warped_piano)
                    # Detect dots in the perspective-corrected image
                    white_dots, black_dots = detect_dots(warped_piano, white_keys, black_keys)
            else:
                corners, rect, transform_matrix, warped_piano = None, None, None, None
        else:
            # Use existing stabilized values
            corners = stabilized_corners
            rect = stabilized_sheet
            transform_matrix = stabilized_perspective_transform
            
            # Get perspective-corrected image using stabilized transform
            warped_piano, _ = apply_perspective_correction(image, stabilized_corners)
            
            # Detect piano keys and dots in the stabilized warped image
            white_keys, black_keys = detect_keys_from_sheet(warped_piano)
            white_dots, black_dots = detect_dots(warped_piano, white_keys, black_keys)
        
        # Display stabilization progress
        if not is_stabilized and stabilization_counter > 0:
            progress = min(100, int(stabilization_counter / MAX_STABILIZATION_FRAMES * 100))
            cv2.putText(image, f"Stabilizing piano... {progress}%", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw the piano sheet outline if detected
        if corners is not None:
            # Draw the quad outline
            for i in range(4):
                cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
        
        # Create a blend of the original image with the detected piano sheet
        if warped_piano is not None:
            # Create a separate visualization window for the corrected piano view
            vis_piano = warped_piano.copy()
            
            # Draw white keys on the warped image
            for wx, wy, ww, wh in white_keys:
                cv2.rectangle(vis_piano, (wx, wy), (wx + ww, wy + wh), (255, 0, 0), 2)
            
            # Draw black keys on the warped image
            for bx, by, bw, bh in black_keys:
                cv2.rectangle(vis_piano, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            
            # Draw dots on the warped image
            for wx, wy, key_idx in white_dots:
                cv2.circle(vis_piano, (wx, wy), 5, (255, 255, 255), -1)
                # Add note name near the dot
                if key_idx in NOTES:
                    cv2.putText(vis_piano, NOTES[key_idx], (wx - 10, wy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for bx, by, key_idx in black_dots:
                cv2.circle(vis_piano, (bx, by), 5, (0, 0, 0), -1)
                # Add note name near the dot
                if key_idx in NOTES:
                    cv2.putText(vis_piano, NOTES[key_idx], (bx - 10, by - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.imshow('Corrected Piano View', vis_piano)
        
        # Track currently pressed keys
        currently_pressed_keys = set()
        
        # Process hand landmarks only after stabilization
        if is_stabilized and results.multi_hand_landmarks and transform_matrix is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on original image
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Detect finger presses if piano sheet detected
                if warped_piano is not None and (white_dots or black_dots):
                    # Get finger tip landmarks
                    h, w, _ = image.shape
                    finger_tips = [
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    ]
                    
                    for i, landmark in enumerate(finger_tips):
                        # Convert normalized coordinates to pixel coordinates in original image
                        orig_x, orig_y = int(landmark.x * w), int(landmark.y * h)
                        
                        # Transform finger position to the perspective-corrected space
                        warped_x, warped_y = transform_point((orig_x, orig_y), transform_matrix)
                        
                        # Draw finger position on both images
                        cv2.circle(image, (orig_x, orig_y), 8, (0, 255, 255), -1)
                        if 0 <= warped_x < canvas_width and 0 <= warped_y < canvas_height:
                            cv2.circle(vis_piano, (warped_x, warped_y), 8, (0, 255, 255), -1)
                        
                        # Check if finger is pressing white dots (on black keys) in warped space
                        is_pressed, key_idx = is_key_pressed((warped_x, warped_y), white_dots)
                        if is_pressed and key_idx is not None:
                            currently_pressed_keys.add(key_idx)
                            # Draw a highlight around the pressed key
                            if key_idx >= 7 and key_idx - 7 < len(black_keys):
                                bx, by, bw, bh = black_keys[key_idx - 7]
                                cv2.rectangle(vis_piano, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
                        
                        # Check if finger is pressing black dots (on white keys) in warped space
                        is_pressed, key_idx = is_key_pressed((warped_x, warped_y), black_dots)
                        if is_pressed and key_idx is not None:
                            currently_pressed_keys.add(key_idx)
                            # Draw a highlight around the pressed key
                            if key_idx < len(white_keys):
                                wx, wy, ww, wh = white_keys[key_idx]
                                cv2.rectangle(vis_piano, (wx, wy), (wx + ww, wy + wh), (0, 255, 255), 3)
        
        # Play sounds for newly pressed keys (only after stabilized)
        if is_stabilized:
            for key in currently_pressed_keys - previously_pressed_keys:
                if key in sounds and (time.time() - last_press_time.get(key, 0)) > 0.3:  # Debounce
                    sounds[key].play()
                    last_press_time[key] = time.time()
                    print(f"Playing note: {NOTES.get(key, 'Unknown')}")
        
        # Update previously pressed keys
        previously_pressed_keys = currently_pressed_keys
        
        # Add status text
        if is_stabilized:
            cv2.putText(image, "Piano Detection Active - Press 'q' to quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Hold piano sheet steady to calibrate", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Piano Detection', image)
        
        # Exit on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()