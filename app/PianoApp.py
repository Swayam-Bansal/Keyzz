# app/PianoApp.py
import cv2
import time
import numpy as np
import logging

import mediapipe as mp

# Import our custom modules
from audio_manager import AudioManager
from perspective_transformer import PerspectiveTransformer
from stabilizer import Stabilizer
from piano_detector import PianoDetector
from visualizer import Visualizer
from game_manager import GameManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class PianoApp:
    """
    Main application that orchestrates video capture, piano sheet detection,
    stabilization, hand landmark processing, interactive key detection, and game logic.
    """
    def __init__(self):
        self.canvas_width = 700
        self.canvas_height = 200
        self.audio_manager = AudioManager()
        self.detector = PianoDetector()
        self.transformer = PerspectiveTransformer(self.canvas_width, self.canvas_height)
        self.stabilizer = Stabilizer(max_frames=30)
        self.visualizer = Visualizer(self.canvas_width, self.canvas_height)
        self.game_manager = GameManager()  # Load our basic song
        self.game_started = False

        self.prev_pressed_keys = set()
        self.freeze_calibration = False
        self.frozen_corners = None
        self.frozen_rect = None
        self.frozen_transform = None

        # Setup MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def run(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logging.error("Error: Could not open video source.")
            return

        logging.info("Camera initialized. Warming up...")
        time.sleep(2.0)
        cv2.namedWindow('Piano Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Corrected Piano View', cv2.WINDOW_NORMAL)

        last_frame_time = time.time()

        # Cached key/dot detection results if calibration is frozen.
        stabilized_white_keys = None
        stabilized_black_keys = None
        stabilized_white_dots = None
        stabilized_black_dots = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Handle calibration freeze
            if self.freeze_calibration:
                stable_corners = self.frozen_corners
                stable_rect = self.frozen_rect
                stable_transform = self.frozen_transform
                warped_image, _ = self.transformer.apply_transform(frame, self.frozen_corners)
                transform_matrix = stable_transform
            else:
                current_corners, current_rect = self.detector.detect_sheet(frame)
                if current_corners is not None and current_rect is not None:
                    stable_corners, stable_rect, stable_transform, warped_image = self.stabilizer.update(
                        frame, current_corners, current_rect, self.transformer)
                    transform_matrix = stable_transform
                    if self.stabilizer.is_stable:
                        self.freeze_calibration = True
                        self.frozen_corners = stable_corners.copy()
                        self.frozen_rect = stable_rect
                        self.frozen_transform = stable_transform.copy()
                        logging.info("Calibration frozen. Using fixed piano contour.")
                else:
                    transform_matrix = None
                    warped_image = None

            # Key and dot detection on warped image
            white_keys, black_keys = [], []
            white_dots, black_dots = [], []
            if warped_image is not None:
                if self.freeze_calibration and stabilized_white_keys is not None:
                    white_keys, black_keys = stabilized_white_keys, stabilized_black_keys
                    white_dots, black_dots = stabilized_white_dots, stabilized_black_dots
                else:
                    white_keys, black_keys = self.detector.detect_keys(warped_image)
                    white_dots, black_dots = self.detector.detect_dots(warped_image, white_keys, black_keys)
                    if self.freeze_calibration:
                        stabilized_white_keys = white_keys
                        stabilized_black_keys = black_keys
                        stabilized_white_dots = white_dots
                        stabilized_black_dots = black_dots

            vis_piano = self.visualizer.draw_piano_view(warped_image, white_keys, black_keys, white_dots, black_dots)
            
            pressed_keys = set()
            finger_points = []
            if results.multi_hand_landmarks and transform_matrix is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )
                    finger_tips = [
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP],
                    ]
                    for landmark in finger_tips:
                        orig_x, orig_y = int(landmark.x * w), int(landmark.y * h)
                        transformed = self.transformer.transform_point((orig_x, orig_y), transform_matrix)
                        if transformed is not None:
                            warped_x, warped_y = transformed
                            cv2.circle(frame, (orig_x, orig_y), 5, (255, 255, 0), -1)
                            if 0 <= warped_x < self.canvas_width and 0 <= warped_y < self.canvas_height:
                                cv2.circle(vis_piano, (warped_x, warped_y), 5, (255, 255, 0), -1)
                            finger_points.append((warped_x, warped_y))
                            key_idx = self.detector.detect_pressed_key((warped_x, warped_y), white_keys, black_keys, margin=0.05)
                            if key_idx is not None:
                                pressed_keys.add(key_idx)
                                if key_idx < 7 and key_idx < len(white_keys):
                                    vis_piano = self.visualizer.highlight_key(vis_piano, white_keys[key_idx])
                                elif key_idx >= 7:
                                    # For black keys, map visual index accordingly.
                                    for visual_idx, n_idx in self.detector.BLACK_KEY_VISUAL_MAP.items():
                                        if n_idx == key_idx and visual_idx < len(black_keys):
                                            vis_piano = self.visualizer.highlight_key(vis_piano, black_keys[visual_idx])
                                            break

            # Play sound for new key presses
            for key in pressed_keys - self.prev_pressed_keys:
                self.audio_manager.play_sound(key)
            self.prev_pressed_keys = pressed_keys

            # Start the game when calibration is frozen
            if self.freeze_calibration and not self.game_started:
                self.game_manager.start_game()
                self.game_started = True

            # Game update and drawing
            current_frame_time = time.time()
            dt = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            # In PianoApp.run() - Find the section where you update the score display
            if self.game_started:
                self.game_manager.update(dt, white_keys, black_keys, pressed_keys)
                self.game_manager.draw_notes(vis_piano, white_keys, black_keys)
                
                # Enhanced display with combo and mistakes
                score = self.game_manager.get_score()
                combo = self.game_manager.get_combo()
                mistakes_left = self.game_manager.get_mistakes_left()
                
                # Display score and combo
                cv2.putText(vis_piano, f"Score: {score}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_piano, f"Combo: {combo}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display mistakes left
                mistake_text = f"Lives: {mistakes_left}"
                mistake_color = (0, 255, 0) if mistakes_left > 1 else (0, 0, 255)
                cv2.putText(vis_piano, mistake_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mistake_color, 2)
                
                # Check if game is over
                if self.game_manager.is_game_over():
                    cv2.putText(vis_piano, "GAME OVER!", (vis_piano.shape[1]//2 - 100, vis_piano.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    cv2.putText(vis_piano, "Press 'r' to restart", (vis_piano.shape[1]//2 - 80, vis_piano.shape[0]//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    max_combo = self.game_manager.get_max_combo()
                    cv2.putText(vis_piano, f"Max Combo: {max_combo}", (vis_piano.shape[1]//2 - 80, vis_piano.shape[0]//2 + 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display status on the main frame.
            if self.freeze_calibration:
                status = "Status: Calibrated (Frozen) - Playing"
                status_color = (0, 255, 0)
            elif self.stabilizer.counter > 0:
                progress = min(100, int(self.stabilizer.counter / self.stabilizer.max_frames * 100))
                status = f"Status: Stabilizing... {progress}%"
                status_color = (0, 255, 255)
            else:
                status = "Status: Searching for Piano Sheet..."
                status_color = (0, 0, 255)
            frame = self.visualizer.draw_status(frame, status, status_color)
            if self.freeze_calibration:
                frame = self.visualizer.draw_detection_outline(frame, self.frozen_corners)
            else:
                frame = self.visualizer.draw_detection_outline(frame, current_corners)
            
            cv2.imshow('Piano Detection', frame)
            cv2.imshow('Corrected Piano View', vis_piano)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset calibration and game state
                self.freeze_calibration = False
                self.stabilizer.counter = 0
                self.stabilizer.is_stable = False
                self.frozen_corners = None
                self.frozen_rect = None
                self.frozen_transform = None
                self.game_started = False

        logging.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.audio_manager  # Ensure Pygame quits automatically when process exits.
        logging.info("Done.")

if __name__ == "__main__":
    app = PianoApp()
    app.run()
