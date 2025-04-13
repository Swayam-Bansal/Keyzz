import cv2
import numpy as np
import logging

# Configure logging for detailed runtime information.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ----------------------------
# Perspective Transformer
# ----------------------------
class PerspectiveTransformer:
    """
    Applies perspective correction to images and transforms points between the
    original and warped coordinate spaces.
    """

    def __init__(self, canvas_width: int = 700, canvas_height: int = 200):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def apply_transform(self, image: np.ndarray, corners: np.ndarray):
        """
        Given an image and 4 ordered corners, computes a perspective transform,
        returning the warped image and the transformation matrix.
        """
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
        """
        Apply the perspective transformation matrix to a point.
        """
        if matrix is None:
            return None
        src_point = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(src_point, matrix)
        if transformed is None or transformed.size == 0 or transformed.shape != (1, 1, 2):
            return None
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))

    def inverse_transform_point(self, point: tuple, matrix: np.ndarray):
        """
        Convert a point from warped image space back to the original image space.
        """
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