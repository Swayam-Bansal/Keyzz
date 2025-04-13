# app/piano_detector.py
import cv2
import numpy as np
import logging

class PianoDetector:
    """
    Detects the piano sheet, segments the keys, and detects key-specific dots
    used for interactive key-press detection.
    """

    # ROI ratios for dot detection
    BLACK_KEY_DOT_ROI_Y_START = 0.1
    BLACK_KEY_DOT_ROI_Y_END = 0.4
    WHITE_KEY_DOT_ROI_Y_START = 0.05
    WHITE_KEY_DOT_ROI_Y_END = 0.3
    BLACK_KEY_NOTE_INDICES = [7, 8, 9, 10, 11]
    BLACK_KEY_VISUAL_MAP = {i: note for i, note in enumerate(BLACK_KEY_NOTE_INDICES)}

    def detect_sheet(self, image):
        """Detect the four corners of the piano sheet in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            for contour in candidates:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4:
                    corners = np.array([point[0] for point in approx])
                    rect = cv2.boundingRect(contour)
                    sums = corners.sum(axis=1)
                    diffs = np.diff(corners, axis=1)
                    ordered_corners = np.zeros((4, 2), dtype=np.int32)
                    ordered_corners[0] = corners[np.argmin(sums)]
                    ordered_corners[2] = corners[np.argmax(sums)]
                    ordered_corners[1] = corners[np.argmin(diffs)]
                    ordered_corners[3] = corners[np.argmax(diffs)]
                    sorted_y = corners[np.argsort(corners[:, 1])]
                    top_corners = sorted_y[:2]
                    bottom_corners = sorted_y[2:]
                    top_sorted = top_corners[np.argsort(top_corners[:, 0])]
                    bottom_sorted = bottom_corners[np.argsort(bottom_corners[:, 0])]
                    ordered_corners_robust = np.array([top_sorted[0], top_sorted[1],
                                                       bottom_sorted[1], bottom_sorted[0]], dtype=np.int32)
                    return ordered_corners_robust, cv2.boundingRect(contour)
            if candidates and len(candidates[0]) >= 4:
                peri = cv2.arcLength(candidates[0], True)
                approx = cv2.approxPolyDP(candidates[0], 0.02 * peri, True)
                if len(approx) == 4:
                    corners = np.array([p[0] for p in approx], dtype=np.int32)
                    sums = corners.sum(axis=1)
                    diffs = np.diff(corners, axis=1)
                    ordered_corners = np.zeros((4, 2), dtype=np.int32)
                    ordered_corners[0] = corners[np.argmin(sums)]
                    ordered_corners[2] = corners[np.argmax(sums)]
                    ordered_corners[1] = corners[np.argmin(diffs)]
                    ordered_corners[3] = corners[np.argmax(diffs)]
                    return ordered_corners, cv2.boundingRect(candidates[0])
        return None, None

    def detect_keys(self, warped_image):
        """Detect white and black keys from the warped image."""
        if warped_image is None or warped_image.size == 0:
            return [], []
        h, w = warped_image.shape[:2]
        num_white_keys = 7
        white_keys = []
        white_key_width = w / num_white_keys
        for i in range(num_white_keys):
            key_x = int(i * white_key_width)
            next_key_x = int((i + 1) * white_key_width)
            white_keys.append((key_x, 0, next_key_x - key_x, h))
        black_keys = []
        black_key_height = int(h * 0.6)
        black_key_width = int(white_key_width * 0.6)
        black_key_start_y = h - black_key_height
        black_key_positions = [0, 1, 3, 4, 5]
        for idx in black_key_positions:
            if idx < num_white_keys - 1:
                boundary_x = (idx + 1) * white_key_width
                black_cx = int(boundary_x)
                black_x = int(black_cx - black_key_width / 2)
                black_keys.append((black_x, black_key_start_y, black_key_width, black_key_height))
        return white_keys, black_keys

    def detect_dots(self, warped_image, white_keys, black_keys):
        """Detect dots on keys for further confirmation of key-press events."""
        if warped_image is None:
            return [], []
        gray_roi = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        h, w = gray_roi.shape
        white_dots, black_dots = [], []
        min_dot_area = 15
        max_dot_area = 250
        block_size = 11
        C_val = 2
        kernel =  np.ones((3, 3), np.uint8)
        block_size_black = 15
        C_val_black = -2
        kernel_black = np.ones((3, 3), np.uint8)
        # Process black keys (white dots)
        for i, (bx, by, bw, bh) in enumerate(black_keys):
            roi_y_start = int(by + bh * self.BLACK_KEY_DOT_ROI_Y_START)
            roi_y_end = int(by + bh * self.BLACK_KEY_DOT_ROI_Y_END)
            roi_x_start, roi_x_end = bx, bx + bw
            roi_y_start = max(0, roi_y_start)
            roi_x_start = max(0, roi_x_start)
            roi_y_end = min(h, roi_y_end)
            roi_x_end = min(w, roi_x_end)
            if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start:
                continue
            key_roi = gray_roi[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            key_roi_eq = cv2.equalizeHist(key_roi)
            black_thresh = cv2.adaptiveThreshold(key_roi_eq, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY,
                                                 block_size_black, C_val_black)
            black_thresh = cv2.morphologyEx(black_thresh, cv2.MORPH_OPEN, kernel_black)
            contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_dot_area < area < max_dot_area:
                    m = cv2.moments(contour)
                    if m["m00"] > 0:
                        cx_roi = int(m["m10"] / m["m00"])
                        cy_roi = int(m["m01"] / m["m00"])
                        cx = cx_roi + roi_x_start
                        cy = cy_roi + roi_y_start
                        note_idx = self.BLACK_KEY_VISUAL_MAP.get(i)
                        if note_idx is not None:
                            white_dots.append((cx, cy, note_idx))
        # Process white keys (dark dots)
        for i, (wx, wy, ww, wh) in enumerate(white_keys):
            roi_y_start = int(wy + wh * self.WHITE_KEY_DOT_ROI_Y_START)
            roi_y_end = int(wy + wh * self.WHITE_KEY_DOT_ROI_Y_END)
            roi_x_start, roi_x_end = wx, wx + ww
            roi_y_start = max(0, roi_y_start)
            roi_x_start = max(0, roi_x_start)
            roi_y_end = min(h, roi_y_end)
            roi_x_end = min(w, roi_x_end)
            if roi_y_end <= roi_y_start or roi_x_end <= roi_x_start:
                continue
            key_roi = gray_roi[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            black_thresh = cv2.adaptiveThreshold(key_roi, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV,
                                                 block_size, C_val)
            black_thresh = cv2.morphologyEx(black_thresh, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_dot_area < area < max_dot_area:
                    m = cv2.moments(contour)
                    if m["m00"] > 0:
                        cx_roi = int(m["m10"] / m["m00"])
                        cy_roi = int(m["m01"] / m["m00"])
                        cx = cx_roi + roi_x_start
                        cy = cy_roi + roi_y_start
                        black_dots.append((cx, cy, i))
        return white_dots, black_dots

    def detect_pressed_key(self, fingertip, white_keys, black_keys, margin: float = 0.05):
        """
        Given a fingertip coordinate on the warped image, determine which key is pressed.
        Returns the note index (as used in the audio manager) if a hit is detected.
        """
        candidates = []
        for idx, (x, y, w, h) in enumerate(white_keys):
            inner_x = x + w * margin
            inner_y = y + h * margin
            inner_w = w * (1 - 2 * margin)
            inner_h = h * (1 - 2 * margin)
            if inner_x <= fingertip[0] <= inner_x + inner_w and inner_y <= fingertip[1] <= inner_y + inner_h:
                center = (x + w / 2, y + h / 2)
                dist = np.hypot(fingertip[0] - center[0], fingertip[1] - center[1])
                candidates.append((dist, idx))
        for visual_idx, (x, y, w, h) in enumerate(black_keys):
            inner_x = x + w * margin
            inner_y = y + h * margin
            inner_w = w * (1 - 2 * margin)
            inner_h = h * (1 - 2 * margin)
            if inner_x <= fingertip[0] <= inner_x + inner_w and inner_y <= fingertip[1] <= inner_y + inner_h:
                center = (x + w / 2, y + h / 2)
                dist = np.hypot(fingertip[0] - center[0], fingertip[1] - center[1])
                note_idx = self.BLACK_KEY_VISUAL_MAP.get(visual_idx)
                if note_idx is not None:
                    candidates.append((dist, note_idx))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates[0][1]
        return None
