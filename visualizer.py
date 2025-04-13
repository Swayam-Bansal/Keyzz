import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import logging

# Configure logging for detailed runtime information.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# ----------------------------
# Visualizer
# ----------------------------
class Visualizer:
    """
    Handles drawing of the piano keys, dots, status text, and overlays on both the original and warped images.
    """

    def __init__(self, canvas_width: int = 700, canvas_height: int = 200):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def draw_piano_view(self, warped_image: np.ndarray, white_keys: list, black_keys: list,
                          white_dots: list, black_dots: list, finger_points: list = None):
        """
        Create a visualization overlay for the warped piano view.
        """
        if warped_image is None:
            canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "No Piano Detected", (self.canvas_width // 2 - 100, self.canvas_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            return canvas

        canvas = warped_image.copy()
        # Draw white keys.
        for wx, wy, ww, wh in white_keys:
            cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), (220, 220, 220), -1)
            cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), (180, 180, 180), 1)
        # Draw black keys.
        for bx, by, bw, bh in black_keys:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        # Draw dots.
        dot_radius = 6
        for dx, dy, _ in white_dots:
            cv2.circle(canvas, (dx, dy), dot_radius, (255, 255, 255), -1)
            cv2.circle(canvas, (dx, dy), dot_radius, (0, 0, 0), 1)
        for dx, dy, _ in black_dots:
            cv2.circle(canvas, (dx, dy), dot_radius, (0, 0, 0), -1)
            cv2.circle(canvas, (dx, dy), dot_radius, (255, 255, 255), 1)
        # Optionally draw finger points.
        if finger_points:
            for pt in finger_points:
                cv2.circle(canvas, pt, 5, (255, 255, 0), -1)
        return canvas

    def draw_status(self, image: np.ndarray, status_text: str, status_color: tuple):
        """Overlay status text on the image."""
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return image

    def draw_detection_outline(self, image: np.ndarray, corners: np.ndarray):
        """Draw the outline of the detected piano sheet on the image."""
        if corners is not None:
            cv2.polylines(image, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        return image

    def highlight_key(self, canvas: np.ndarray, key_rect: tuple):
        """Highlight a detected pressed key."""
        x, y, w, h = key_rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return canvas