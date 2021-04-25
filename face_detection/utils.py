import pathlib
from itertools import chain
import cv2


def parse_file(base_path: str, f_format: list):
    base = pathlib.Path(base_path)
    if base.exists():
        for f in chain(f_format):
            for p in base.rglob('*.' + f):
                yield str(p)
    else:
        return None


def draw_face(frame, pt_1, pt_2, r, d, color, thickness):
    x1, y1 = pt_1
    x2, y2 = pt_2

    cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    return frame
