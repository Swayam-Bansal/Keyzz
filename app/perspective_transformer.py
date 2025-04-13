# app/perspective_transformer.py
import cv2
import numpy as np
import logging

class PerspectiveTransformer:
    """Applies perspective correction and provides point transformation utilities."""
    def __init__(self, canvas_width: int = 700, canvas_height: int = 200):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def apply_transform(self, image: np.ndarray, corners: np.ndarray):
        """Given an image and 4 ordered corners, compute and apply perspective transform."""
        if corners is None or len(corners) != 4:
            return None, None
        dst_corners = np.array([
            [0, 0],
            [self.canvas_width, 0],
            [self.canvas_width, self.canvas_height],
            [0, self.canvas_height]
        ], dtype=np.float32)
        src_corners = corners.astype(np.float32)
        try:
            matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
            warped = cv2.warpPerspective(image, matrix, (self.canvas_width, self.canvas_height))
            return warped, matrix
        except cv2.error as e:
            logging.error(f"Perspective transform error: {e}")
            return None, None

    def transform_point(self, point: tuple, matrix: np.ndarray):
        """Transform a point from the original image space to warped space."""
        if matrix is None:
            return None
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(src_point, matrix)
        if transformed is None or transformed.size == 0 or transformed.shape != (1, 1, 2):
            return None
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def inverse_transform_point(self, point: tuple, matrix: np.ndarray):
        """Transform a point from warped space back to the original image space."""
        if matrix is None:
            return None
        retval, inv_matrix = cv2.invert(matrix)
        if not retval:
            logging.error("Failed to invert perspective matrix.")
            return None
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(src_point, inv_matrix)
        if transformed is None or transformed.size == 0 or transformed.shape != (1, 1, 2):
            return None
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
