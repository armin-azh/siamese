import pathlib
from itertools import chain
import cv2
import numpy as np


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


def iou(bb_test, bb_gt):
    """
    Computes IOU between two boxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2]+bb_test[0], bb_gt[2]+bb_gt[0])
    yy2 = np.minimum(bb_test[3]+bb_test[1], bb_gt[3]+bb_gt[1])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]) * (bb_test[3])
              + (bb_gt[2]) * (bb_gt[3]) - wh)
    return o


def bulk_calculate_iou(boxes1, boxes2):
    """
    calculate iou of m sample against n instances
    :param boxes1:
    :param boxes2:
    :return:
    """
    box1_size = boxes1.shape[0]
    box2_size = boxes2.shape[0]
    iou_matrix = np.zeros((box1_size, box2_size))
    for i in range(box1_size):
        for j in range(box2_size):
            iou_matrix[i, j] = iou(boxes1[i, :], boxes2[j, :])

    return iou_matrix
