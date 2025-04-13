import audio_manager
import perspective_transformer
import stabilizer
import piano_detector
import visualizer

import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import logging

# Configure logging for detailed runtime information.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')



# ----------------------------
# Main Piano Application
# ----------------------------
class PianoApp:
    """
    Orchestrates video capture, piano sheet detection, stabilization, hand landmark processing,
    key-press detection, audio playback, and visualization.
    
    New Feature: Calibration Freeze – Once stabilization is reached, the contour is frozen.
    """
    def __init__(self):
        self.canvas_width = 700
        self.canvas_height = 200
        self.audio_manager = audio_manager.AudioManager()
        self.detector = piano_detector.PianoDetector()
        self.transformer = perspective_transformer.PerspectiveTransformer(self.canvas_width, self.canvas_height)
        self.stabilizer = stabilizer.Stabilizer(max_frames=30)
        self.visualizer = visualizer.Visualizer(self.canvas_width, self.canvas_height)
        self.prev_pressed_keys = set()
        # Calibration freeze attributes.
        self.freeze_calibration = False
        self.frozen_corners = None
        self.frozen_rect = None
        self.frozen_transform = None

        # Initialize MediaPipe Hands.
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

        logging.info("Camera initialized. Waiting for warm-up...")
        time.sleep(2.0)
        cv2.namedWindow('Piano Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Corrected Piano View', cv2.WINDOW_NORMAL)

        stabilized_white_keys = None
        stabilized_black_keys = None
        stabilized_white_dots = None
        stabilized_black_dots = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logging.warning("Failed to capture frame; retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Freeze calibration: if already frozen, use stored values.
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
                    # Freeze calibration when stabilization completes.
                    if self.stabilizer.is_stable:
                        self.freeze_calibration = True
                        self.frozen_corners = stable_corners.copy()
                        self.frozen_rect = stable_rect
                        self.frozen_transform = stable_transform.copy()
                        logging.info("Calibration frozen. Using fixed piano contour.")
                else:
                    if not self.freeze_calibration:
                        logging.info("No valid detection; waiting for calibration...")
                        transform_matrix = None
                        warped_image = None
                        stabilized_white_keys = None
                        stabilized_black_keys = None
                        stabilized_white_dots = None
                        stabilized_black_dots = None

            # Key and dot detection.
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
                all_dots = white_dots + black_dots
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
                                    for visual_idx, n_idx in self.detector.BLACK_KEY_VISUAL_MAP.items():
                                        if n_idx == key_idx and visual_idx < len(black_keys):
                                            vis_piano = self.visualizer.highlight_key(vis_piano, black_keys[visual_idx])
                                            break

            for key in pressed_keys - self.prev_pressed_keys:
                self.audio_manager.play_sound(key)
            self.prev_pressed_keys = pressed_keys

            # Status text based on calibration state.
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
            frame = self.visualizer.draw_detection_outline(frame, self.frozen_corners if self.freeze_calibration else current_corners)
            cv2.imshow('Piano Detection', frame)
            cv2.imshow('Corrected Piano View', vis_piano)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                logging.info("Manual reset triggered. Clearing calibration freeze.")
                self.freeze_calibration = False
                self.stabilizer.counter = 0
                self.stabilizer.is_stable = False
                self.frozen_corners = None
                self.frozen_rect = None
                self.frozen_transform = None
                stabilized_white_keys = None
                stabilized_black_keys = None
                stabilized_white_dots = None
                stabilized_black_dots = None

        logging.info("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        pygame.mixer.quit()
        logging.info("Done.")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    app = PianoApp()
    app.run()