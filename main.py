import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Define piano notes (C4 to B4 for simplicity)
NOTES = {
    0: 'C4'
    #   1: 'D4', 2: 'E4', 3: 'F4', 4: 'G4', 5: 'A4', 6: 'B4',
    # 7: 'C#4', 8: 'D#4', 9: 'F#4', 10: 'G#4', 11: 'A#4'
}

# Load sound files (placeholder for actual sound files)
sounds = {}
for key, note in NOTES.items():
    try:
        sounds[key] = pygame.mixer.Sound(f"sounds/{note}.wav")
    except:
        print(f"Warning: Sound file for {note} not found. Creating placeholder.")
        sounds[key] = pygame.mixer.Sound(pygame.sndarray.make_sound(np.zeros((8000,), dtype=np.int16)))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_piano_sheet(image):
    """Detect the sheet with printed piano in the image"""
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
            # Check if contour is rectangular (piano sheet likely is)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # If we have a rectangle
                rect = cv2.boundingRect(approx)
                x, y, w, h = rect
                # Check if the aspect ratio is reasonable for a piano keyboard
                aspect_ratio = w / float(h)
                if 1.2 < aspect_ratio < 5.0:  # Piano keyboard usually wider than tall
                    return rect
        
        # If no suitable rectangle found, just return the largest contour
        return cv2.boundingRect(candidates[0])
    
    return None

def detect_keys_from_sheet(image, sheet_rect):
    """Detect piano keys from the printed piano sheet - with inverted orientation"""
    if sheet_rect is None:
        return [], []
    
    x, y, w, h = sheet_rect
    
    # Define the number of white keys
    num_white_keys = 7
    
    # Create white keys (blue in visualization)
    white_keys = []
    white_key_width = w // num_white_keys
    
    for i in range(num_white_keys):
        key_x = x + i * white_key_width
        white_keys.append((key_x, y, white_key_width, h))
    
    # Create black keys (red in visualization)
    # In inverted orientation, black keys should be at the BOTTOM part
    # Standard piano has black keys at positions 0, 1, 3, 4, 5 (when counting spaces between white keys)
    black_keys = []
    black_key_width = int(white_key_width * 0.6)
    black_key_height = int(h * 0.6)
    black_key_positions = [0, 1, 3, 4, 5]  # C#, D#, F#, G#, A#
    
    for pos in black_key_positions:
        black_x = x + (pos * white_key_width) + (white_key_width * 0.7)
        # Important change: Position black keys at BOTTOM of white keys instead of top
        black_y = y + h - black_key_height  # Position at bottom of white keys
        black_keys.append((int(black_x - black_key_width/2), black_y, black_key_width, black_key_height))
    
    return white_keys, black_keys

def detect_dots(image, sheet_rect, white_keys, black_keys):
    """Detect the white and black dots on the piano sheet with inverted orientation"""
    if sheet_rect is None:
        return [], []
    
    x, y, w, h = sheet_rect
    roi = image[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Lists to store dots with their key indices
    white_dots = []  # Dots on black keys (will be white dots)
    black_dots = []  # Dots on white keys (will be black dots)
    
    # Process black keys to find white dots
    for i, (bx, by, bw, bh) in enumerate(black_keys):
        # Extract ROI for this black key
        key_x1 = max(0, bx - x)
        key_x2 = min(w, bx + bw - x)
        key_y1 = max(0, by - y)
        key_y2 = min(h, by + bh - y)
        
        if key_x1 >= key_x2 or key_y1 >= key_y2:
            continue
            
        key_roi = gray_roi[key_y1:key_y2, key_x1:key_x2]
        
        if key_roi.size == 0:
            continue
            
        # Threshold to find white dots on black keys
        _, white_thresh = cv2.threshold(key_roi, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in white_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 150:  # Adjust threshold based on your dot size
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"]) + key_x1 + x
                    cy = int(m["m01"] / m["m00"]) + key_y1 + y
                    # Store the dot with its corresponding black key index
                    white_dots.append((cx, cy, i + 7))  # Offset by 7 for black keys
    
    # Process white keys to find black dots
    for i, (wx, wy, ww, wh) in enumerate(white_keys):
        # For inverted layout, check upper half of white keys for dots (since black keys are at bottom)
        key_x1 = max(0, wx - x)
        key_x2 = min(w, wx + ww - x)
        key_y1 = max(0, wy - y)  # Start from top of key
        key_y2 = min(h, wy + wh//2 - y)  # Only examine top half
        
        if key_x1 >= key_x2 or key_y1 >= key_y2:
            continue
            
        key_roi = gray_roi[key_y1:key_y2, key_x1:key_x2]
        
        if key_roi.size == 0:
            continue
            
        # Threshold to find black dots on white keys
        _, black_thresh = cv2.threshold(key_roi, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in black_contours:
            area = cv2.contourArea(contour)
            if 10 < area < 150:  # Adjust threshold based on your dot size
                m = cv2.moments(contour)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"]) + key_x1 + x
                    cy = int(m["m01"] / m["m00"]) + key_y1 + y
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

def detect_finger_presses(hand_landmarks, image_shape, white_dots, black_dots):
    """Detect if fingers are pressing on dots"""
    if not hand_landmarks:
        return []
    
    pressed_keys = []
    h, w, _ = image_shape
    
    # Get finger tip landmarks
    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    for finger_id, landmark in enumerate(finger_tips):
        # Convert normalized coordinates to pixel coordinates
        finger_x, finger_y = int(landmark.x * w), int(landmark.y * h)
        
        # Check white dots (on black keys)
        is_pressed, key_idx = is_key_pressed((finger_x, finger_y), white_dots)
        if is_pressed and key_idx is not None:
            pressed_keys.append(key_idx)
        
        # Check black dots (on white keys)
        is_pressed, key_idx = is_key_pressed((finger_x, finger_y), black_dots)
        if is_pressed and key_idx is not None:
            pressed_keys.append(key_idx)
    
    return pressed_keys

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
        
        # Detect the piano sheet
        sheet_rect = detect_piano_sheet(image)
        
        # Initialize dots
        white_dots, black_dots = [], []
        white_keys, black_keys = [], []
        
        if sheet_rect is not None:
            x, y, w, h = sheet_rect
            # Draw rectangle around the detected piano sheet
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Detect piano keys with inverted orientation
            white_keys, black_keys = detect_keys_from_sheet(image, sheet_rect)
            
            # Draw white keys (in blue)
            for wx, wy, ww, wh in white_keys:
                cv2.rectangle(image, (wx, wy), (wx + ww, wy + wh), (255, 0, 0), 2)
            
            # Draw black keys (in red)
            for bx, by, bw, bh in black_keys:
                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            
            # Detect dots with inverted orientation
            white_dots, black_dots = detect_dots(image, sheet_rect, white_keys, black_keys)
            
            # Draw dots
            for wx, wy, _ in white_dots:
                cv2.circle(image, (wx, wy), 5, (255, 255, 255), -1)
            
            for bx, by, _ in black_dots:
                cv2.circle(image, (bx, by), 5, (0, 0, 0), -1)
            
            # Draw note names
            for i, (wx, wy, ww, wh) in enumerate(white_keys):
                if i < len(NOTES):
                    note_name = NOTES[i]
                    # Position text at the bottom of white keys
                    cv2.putText(image, note_name, (wx + ww//2 - 10, wy + wh - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            for i, (bx, by, bw, bh) in enumerate(black_keys):
                if i + 7 < len(NOTES):  # Offset for black keys
                    note_name = NOTES[i + 7]
                    # Position text at the middle of black keys
                    cv2.putText(image, note_name, (bx + bw//2 - 10, by + bh//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Track currently pressed keys
        currently_pressed_keys = set()
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Detect finger presses if piano sheet detected
                if sheet_rect is not None and (white_dots or black_dots):
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
                        # Convert normalized coordinates to pixel coordinates
                        finger_x, finger_y = int(landmark.x * w), int(landmark.y * h)
                        
                        # Draw finger position
                        cv2.circle(image, (finger_x, finger_y), 8, (0, 255, 255), -1)
                        
                        # Check if finger is pressing white dots (on black keys)
                        is_pressed, key_idx = is_key_pressed((finger_x, finger_y), white_dots)
                        if is_pressed and key_idx is not None:
                            currently_pressed_keys.add(key_idx)
                            # Draw a highlight around the pressed key
                            if key_idx >= 7 and key_idx - 7 < len(black_keys):
                                bx, by, bw, bh = black_keys[key_idx - 7]
                                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
                        
                        # Check if finger is pressing black dots (on white keys)
                        is_pressed, key_idx = is_key_pressed((finger_x, finger_y), black_dots)
                        if is_pressed and key_idx is not None:
                            currently_pressed_keys.add(key_idx)
                            # Draw a highlight around the pressed key
                            if key_idx < len(white_keys):
                                wx, wy, ww, wh = white_keys[key_idx]
                                cv2.rectangle(image, (wx, wy), (wx + ww, wy + wh), (0, 255, 255), 3)
        
        # Play sounds for newly pressed keys
        for key in currently_pressed_keys - previously_pressed_keys:
            if key in sounds and (time.time() - last_press_time.get(key, 0)) > 0.3:  # Debounce
                sounds[key].play()
                last_press_time[key] = time.time()
                print(f"Playing note: {NOTES.get(key, 'Unknown')}")
        
        # Update previously pressed keys
        previously_pressed_keys = currently_pressed_keys
        
        # Add text instruction
        cv2.putText(image, "Press 'q' to quit", (10, 30), 
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