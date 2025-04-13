# app/stabilizer.py
import cv2
import numpy as np
import logging

class Stabilizer:
    """Smooths and stabilizes the detected piano sheet over multiple frames."""
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.counter = 0
        self.is_stable = False
        self.stabilized_corners = None
        self.stabilized_rect = None
        self.stabilized_transform = None

    def update(self, image: np.ndarray, current_corners: np.ndarray, current_rect: tuple, transformer):
        """
        Update the stabilized detection.
        Returns: (stabilized_corners, stabilized_rect, stabilized_transform, warped_image)
        """
        if current_corners is None or current_rect is None:
            self.counter = 0
            self.is_stable = False
            self.stabilized_corners = None
            self.stabilized_rect = None
            self.stabilized_transform = None
            logging.info("Detection lost; resetting stabilization.")
            return None, None, None, None

        warped, current_transform = transformer.apply_transform(image, current_corners)
        if warped is None or current_transform is None:
            self.counter = 0
            self.is_stable = False
            logging.info("Perspective correction failed; resetting stabilization.")
            return None, None, None, None

        if self.is_stable:
            stable_warp, _ = transformer.apply_transform(image, self.stabilized_corners)
            if stable_warp is None:
                logging.warning("Failed to warp using stabilized transform.")
            return self.stabilized_corners, self.stabilized_rect, self.stabilized_transform, stable_warp

        if self.counter == 0:
            self.stabilized_corners = current_corners.copy()
            self.stabilized_rect = current_rect
            self.stabilized_transform = current_transform.copy()
            self.counter += 1
            logging.info(f"Stabilization started (1/{self.max_frames})")
            return self.stabilized_corners, self.stabilized_rect, self.stabilized_transform, warped

        corner_diffs = np.sqrt(np.sum((current_corners - self.stabilized_corners) ** 2, axis=1))
        avg_corner_diff = np.mean(corner_diffs)
        x1, y1, w1, h1 = current_rect
        x2, y2, w2, h2 = self.stabilized_rect
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        center_distance = np.hypot(center1[0] - center2[0], center1[1] - center2[1])
        area1 = w1 * h1
        area2 = w2 * h2
        area_ratio = area1 / area2 if area2 > 0 else 0

        if avg_corner_diff < 20 and center_distance < 30 and (0.8 < area_ratio < 1.2):
            self.counter += 1
            alpha = 0.1
            smoothed = alpha * current_corners + (1 - alpha) * self.stabilized_corners
            self.stabilized_corners = smoothed.astype(np.int32)
            self.stabilized_rect = current_rect
            warped_smoothed, smoothed_transform = transformer.apply_transform(image, self.stabilized_corners)
            if warped_smoothed is None or smoothed_transform is None:
                logging.warning("Smoothing produced invalid transform; resetting stabilization.")
                self.counter = 0
                self.is_stable = False
                return current_corners, current_rect, current_transform, warped
            self.stabilized_transform = smoothed_transform.copy()
            warped = warped_smoothed
            logging.info(f"Stabilizing... ({self.counter}/{self.max_frames})")
            if self.counter >= self.max_frames:
                self.is_stable = True
                logging.info("Piano sheet detection stabilized!")
            return self.stabilized_corners, self.stabilized_rect, self.stabilized_transform, warped
        else:
            logging.info(f"Detection unstable; resetting stabilization.")
            self.counter = 0
            self.is_stable = False
            self.stabilized_corners = current_corners.copy()
            self.stabilized_rect = current_rect
            self.stabilized_transform = current_transform.copy()
            return current_corners, current_rect, current_transform, warped
