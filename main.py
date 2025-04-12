import cv2
import numpy as np
import mediapipe as mp
import pygame
import time

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Define piano notes (C4 to B4 for simplicity)
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
                if 1.2 < aspect_ratio < 3.0:  # Piano keyboard usually wider than tall
                    return rect
        
        # If no suitable rectangle found, just return the largest contour
        return cv2.boundingRect(candidates[0])
    
    return None

def detect_keys_from_sheet(image, sheet_rect):
    """Detect piano keys from the printed piano sheet"""
    if sheet_rect is None:
        return [], []
    
    x, y, w, h = sheet_rect
    # Extract region of interest (ROI)
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate black and white
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find white keys - assume 7 equal divisions
    white_keys = []
    key_width = w // 7
    
    for i in range(7):
        key_x = i * key_width
        white_keys.append((int(x + key_x), int(y), int(key_width), int(h)))
    
    # Find black keys - based on the typical piano layout
    black_keys = []
    black_key_width = int(key_width * 0.6)
    black_key_height = int(h * 0.6)
    black_positions = [0, 1, 3, 4, 5]  # Positions where black keys appear
    
    for i in black_positions:
        # Position black keys between white keys
        black_x = int(x + (i + 0.7) * key_width)
        black_keys.append((black_x, int(y), black_key_width, black_key_height))
    
    return white_keys, black_keys

def detect_dots(image, sheet_rect):
    """Detect the white and black dots on the piano sheet"""
    if sheet_rect is None:
        return [], []
    
    x, y, w, h = sheet_rect
    roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Use different thresholds for white and black dots
    # For white dots (on black keys)
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # For black dots (on white keys)
    _, black_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process white dots
    white_dots = []
    for contour in white_contours:
        area = cv2.contourArea(contour)
        # Filter by expected dot size - adjust these values based on your actual image
        if 10 < area < 150:
            m = cv2.moments(contour)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"]) + x
                cy = int(m["m01"] / m["m00"]) + y
                # Check if it's on a black key (near the middle of the piano vertically)
                if y + h * 0.2 < cy < y + h * 0.6:
                    white_dots.append((int(cx), int(cy)))
    
    # Process black dots
    black_dots = []
    for contour in black_contours:
        area = cv2.contourArea(contour)
        # Filter by expected dot size
        if 10 < area < 150:
            m = cv2.moments(contour)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"]) + x
                cy = int(m["m01"] / m["m00"]) + y
                # Check if it's on a white key (lower portion of the piano)
                if cy > y + h * 0.6:
                    black_dots.append((int(cx), int(cy)))
    
    return white_dots, black_dots

def is_key_pressed(finger_tip, dots, threshold=30):
    """Check if a finger is pressing near a dot"""
    for i, (dot_x, dot_y) in enumerate(dots):
        distance = np.sqrt((finger_tip[0] - dot_x)**2 + (finger_tip[1] - dot_y)**2)
        if distance < threshold:
            return True, i
    return False, None

def detect_finger_presses(hand_landmarks, image_shape, white_keys, black_keys, white_dots, black_dots):
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
        if is_pressed is not None and is_pressed:
            # Map to the correct black key
            black_key_idx = 7 + key_idx  # Offset for black keys in the NOTES dict
            pressed_keys.append(black_key_idx)
        
        # Check black dots (on white keys)
        is_pressed, key_idx = is_key_pressed((finger_x, finger_y), black_dots)
        if is_pressed is not None and is_pressed:
            # Map directly to the white key
            pressed_keys.append(key_idx)
    
    return pressed_keys

def main():
    # For webcam input
    cap = cv2.VideoCapture(1)
    
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
            
            # Detect piano keys
            white_keys, black_keys = detect_keys_from_sheet(image, sheet_rect)
            
            # Draw white keys
            for wx, wy, ww, wh in white_keys:
                cv2.rectangle(image, (wx, wy), (wx + ww, wy + wh), (255, 0, 0), 2)
            
            # Draw black keys
            for bx, by, bw, bh in black_keys:
                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            
            # Detect dots
            white_dots, black_dots = detect_dots(image, sheet_rect)
            
            # Draw dots
            for wx, wy in white_dots:
                cv2.circle(image, (wx, wy), 5, (255, 255, 255), -1)
            
            for bx, by in black_dots:
                cv2.circle(image, (bx, by), 5, (0, 0, 0), -1)
        
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
                if sheet_rect is not None and white_dots and black_dots:
                    pressed_keys = detect_finger_presses(
                        hand_landmarks, image.shape, white_keys, black_keys, white_dots, black_dots)
                    currently_pressed_keys.update(pressed_keys)
        
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
        
        # image = cv2.flip(image, 1)
        
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