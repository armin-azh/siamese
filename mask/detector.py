import cv2
import numpy as np
from typing import List
import abc


class AbstractMaskDetector:
    def __init__(self, *args, **kwargs):
        super(AbstractMaskDetector, self).__init__(*args, **kwargs)

    def has_mask(self, img: np.ndarray, coordinate: dict) -> bool:
        raise NotImplementedError

    def put_mask(self, img: np.ndarray, coordinate: dict) -> np.ndarray:
        raise NotImplementedError

    def run(self, faces: np.ndarray, coordinates: List[dict]) -> np.ndarray:
        raise NotImplementedError


class LaplacianMask(AbstractMaskDetector):
    def __init__(self, *args, **kwargs):
        super(LaplacianMask, self).__init__(*args, **kwargs)

    def _laplacian_edge_detector(self, img: np.ndarray) -> np.ndarray:

        gray_mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask_edge = cv2.Laplacian(gray_mask, cv2.CV_8U)
        return mask_edge

    def has_mask(self, img: np.ndarray, coordinate: dict) -> bool:

        nose = coordinate.get("nose")
        nose = img[nose[1] - 10: nose[1] + 10, nose[0] - 10: nose[0] + 10, :]
        mouth_l = coordinate.get("mouth_left")
        mouth_r = coordinate.get("mouth_right")
        mouth = img[mouth_r[1] - 10: mouth_r[1] + 10, mouth_l[0]:mouth_r[0], :]
        mouth = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
        nose = cv2.cvtColor(nose, cv2.COLOR_RGB2GRAY)
        av_mouth = np.average(mouth)
        av_nose = np.average(nose)
        av_dist = abs(av_nose - av_mouth)
        th = 10
        if av_dist < th:
            return True
        else:
            return False

    def put_mask(self, img: np.ndarray, coordinate: dict) -> np.ndarray:
        keypoint = coordinate.get("keypoints")

        x, y, w, h = coordinate.get("box")
        right_eye = keypoint.get("right_eye")
        mask_height = y + h - right_eye[1]
        mask_width = x + w
        mask = img[right_eye[1] + 10:right_eye[1] + mask_height, x: mask_width, :]

        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_edge = cv2.Laplacian(gray_mask, cv2.CV_8U)

        mask_edge = np.dstack([mask_edge, mask_edge, mask_edge])
        img[right_eye[1] + 10:right_eye[1] + mask_height, x: mask_width, :] = mask_edge

        return img

    def run(self, faces: np.ndarray, coordinates: List[dict]) -> np.ndarray:
        pass
        # f_ = np.zeros_like(faces)
        #
        # idx = 0
        #
        # for face in faces:
        #     idx += 1
        #     if self.has_mask(img=face, coordinate=coordinates[idx]):
        #         pass
        #
        #     f_[idx] = face
        #
        # return f_
