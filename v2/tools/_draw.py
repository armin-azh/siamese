from typing import Tuple
import cv2
import numpy as np


def draw_face(mat: np.ndarray, pt_1: Tuple[int, int], pt_2: [int, int], r: int, d: int, color: Tuple[int, int, int],
              thickness: int) -> np.ndarray:
    x1, y1 = pt_1
    x2, y2 = pt_2

    cv2.line(mat, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(mat, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(mat, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(mat, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(mat, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(mat, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(mat, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(mat, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(mat, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(mat, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(mat, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(mat, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    return mat
