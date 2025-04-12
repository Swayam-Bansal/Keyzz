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

# Load sound files (you'd need to provide actual sound files)
sounds = {}
for key, note in NOTES.items():
    try:
        sounds[key] = pygame.mixer.Sound(f"sounds/{note}.wav")
    except:
        print(f"Warning: Sound file for {note} not found. Using placeholder.")
        # Create a short silence as placeholder
        sounds[key] = pygame.mixer.Sound(pygame.mixer.Sound(pygame.sndarray.make_sound(np.zeros((500,), dtype=np.int16))))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_piano(image):
    """Detect piano keys in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate black and white
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the piano)
    if contours:
        piano_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(piano_contour)
        return x, y, w, h
    return None

def detect_keys(image, piano_rect):
    """Detect white and black keys within the piano"""
    if piano_rect is None:
        return [], []
        
    x, y, w, h = piano_rect
    piano_roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(piano_roi, cv2.COLOR_BGR2GRAY)
    
    # Extract white keys (they have white dots at the bottom)
    white_keys = []
    key_width = w // 7  # Assuming 7 white keys
    
    for i in range(7):
        key_x = x + i * key_width
        white_keys.append((key_x, y, key_width, h))
    
    # Extract black keys (they have white dots on them)
    black_keys = []
    black_key_width = key_width // 2
    black_key_height = h * 2 // 3
    
    # Positions for the 5 black keys
    black_key_positions = [0, 1, 3, 4, 5]
    
    for pos in black_key_positions:
        if pos < 2:
            key_x = x + (pos + 0.75) * key_width - black_key_width//2
        else:
            key_x = x + (pos + 0.75) * key_width - black_key_width//2
        black_keys.append((key_x, y, black_key_width, black_key_height))
    
    return white_keys, black_keys

def detect_dots(image, piano_rect):
    """Detect the white and black dots on the piano"""
    if piano_rect is None:
        return [], []
        
    x, y, w, h = piano_rect
    piano_roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(piano_roi, cv2.COLOR_BGR2GRAY)
    
    # Detect white dots (on black keys)
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    white_dots = []
    for contour in white_contours:
        area = cv2.contourArea(contour)
        if 20 < area < 200:  # Filter by expected dot size
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"]) + x
                cy = int(m["m01"] / m["m00"]) + y
                white_dots.append((cx, cy))
    
    # Detect black dots (on white keys)
    _, black_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_dots = []
    for contour in black_contours:
        area = cv2.contourArea(contour)
        if 20 < area < 200:  # Filter by expected dot size
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"]) + x
                cy = int(m["m01"] / m["m00"]) + y
                black_dots.append((cx, cy))
    
    return white_dots, black_dots

def is_key_pressed(finger_tip, dots, threshold=30):
    """Check if a finger is pressing near a dot"""
    for dot_x, dot_y in dots:
        distance = np.sqrt((finger_tip[0] - dot_x)**2 + (finger_tip[1] - dot_y)**2)
        if distance < threshold:
            return True, (dot_x, dot_y)
    return False, None

def detect_finger_presses(hand_landmarks, image_shape, white_dots, black_dots):
    """Detect if fingers are pressing on any dots"""
    if not hand_landmarks:
        return []
    
    pressed_keys = []
    
    # Get finger tip landmarks
    h, w, _ = image_shape
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
        
        # Check for white dots (black keys)
        is_pressed, dot = is_key_pressed((finger_x, finger_y), white_dots)
        if is_pressed:
            # Find which black key this is
            key_idx = white_dots.index(dot)
            pressed_keys.append(key_idx + 7)  # Offset for black keys
        
        # Check for black dots (white keys)
        is_pressed, dot = is_key_pressed((finger_x, finger_y), black_dots)
        if is_pressed:
            # Find which white key this is
            key_idx = black_dots.index(dot)
            pressed_keys.append(key_idx)
    
    return pressed_keys

def main():
    # For webcam input (0 for default camera, or provide video file path)
    cap = cv2.VideoCapture(1)
    
    # Variables to track pressed keys
    previously_pressed_keys = set()
    last_press_time = {}
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        # Flip the image horizontally for a mirrored view
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get hand landmarks
        results = hands.process(image_rgb)
        
        # Detect piano
        piano_rect = detect_piano(image)
        if piano_rect:
            x, y, w, h = piano_rect
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Detect keys
            white_keys, black_keys = detect_keys(image, piano_rect)
            
            # Draw white keys
            for key_x, key_y, key_w, key_h in white_keys:
                cv2.rectangle(image, (key_x, key_y), (key_x+key_w, key_y+key_h), (255, 0, 0), 2)
            
            # Draw black keys
            for key_x, key_y, key_w, key_h in black_keys:
                cv2.rectangle(image, (key_x, key_y), (key_x+key_w, key_y+key_h), (0, 0, 255), 2)
            
            # Detect dots
            white_dots, black_dots = detect_dots(image, piano_rect)
            
            # Draw white dots
            for dot_x, dot_y in white_dots:
                cv2.circle(image, (dot_x, dot_y), 5, (255, 255, 255), -1)
            
            # Draw black dots
            for dot_x, dot_y in black_dots:
                cv2.circle(image, (dot_x, dot_y), 5, (0, 0, 0), -1)
        
        # Check for finger presses
        currently_pressed_keys = set()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect finger presses
                if piano_rect and white_dots and black_dots:
                    pressed_keys = detect_finger_presses(
                        hand_landmarks, image.shape, white_dots, black_dots)
                    currently_pressed_keys.update(pressed_keys)
        
        # Play sounds for newly pressed keys
        for key in currently_pressed_keys - previously_pressed_keys:
            if key in sounds and (time.time() - last_press_time.get(key, 0)) > 0.3:  # Debounce
                sounds[key].play()
                last_press_time[key] = time.time()
                print(f"Playing note: {NOTES.get(key, 'Unknown')}")
        
        # Update previously pressed keys
        previously_pressed_keys = currently_pressed_keys
        
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