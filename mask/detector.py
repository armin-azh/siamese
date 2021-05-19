import cv2
import numpy as np
from typing import List
import abc


class AbstractMaskDetector:
    def __init__(self, *args, **kwargs):
        super(AbstractMaskDetector, self).__init__(*args, **kwargs)

    def has_mask(self, img: np.ndarray, coordinate: np.ndarray) -> bool:
        raise NotImplementedError

    def put_mask(self, img: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(self, faces: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LaplacianMask(AbstractMaskDetector):
    def __init__(self, *args, **kwargs):
        super(LaplacianMask, self).__init__(*args, **kwargs)

    def _laplacian_edge_detector(self, img: np.ndarray) -> np.ndarray:

        gray_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_edge = cv2.Laplacian(gray_mask, cv2.CV_8U)
        return mask_edge

    def has_mask(self, img: np.ndarray, coordinate: np.ndarray) -> bool:

        nose = coordinate[4:6]

        nose = img[nose[1] - 10: nose[1] + 10, nose[0] - 10: nose[0] + 10, :]
        mouth_l = coordinate[12:]
        mouth_r = coordinate[10:12]
        mouth = img[mouth_r[1] - 10: mouth_r[1] + 10, mouth_l[0]:mouth_r[0], :]
        mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        nose = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
        av_mouth = np.average(mouth)
        av_nose = np.average(nose)
        av_dist = abs(av_nose - av_mouth)
        th = 10
        if av_dist < th:
            return True
        else:
            return False

    def put_mask(self, img: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
        try:
            x, y = coordinate[:2]
            w, h = coordinate[2:4]
            noise_ = coordinate[4:6]
            # l_mouth = coordinate[12:]
            r_mouth = coordinate[10:12]
            l_eye = coordinate[8:10]
            # r_eye = coordinate[6:8]
            #
            # mid_mouth_x = int((r_mouth[0] + l_mouth[0]) / 2)
            mid_mouth_y = r_mouth[1]

            # mid_eye_x = int((l_eye[0] + r_eye[0]) / 2)
            mid_eye_y = l_eye[1]

            mid_nose_eye_y = int((noise_[1] - mid_eye_y) / 2)

            h_ = (mid_mouth_y - noise_[1]) * 2
            y_ = noise_[1] - mid_nose_eye_y

            mask = img[y_:y_ + h_, x:x + w, :]
            # print(mask)
            mask_edge = self._laplacian_edge_detector(mask)
            #
            mask_edge = np.dstack([mask_edge, mask_edge, mask_edge])
            img[y_:y_ + h_, x:x + w, :] = mask_edge
            return img
        except Warning:

            return img

    def run(self, faces: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        faces = faces.astype(np.uint8)
        f_ = np.zeros_like(faces)

        idx = 0

        for face in faces:

            if self.has_mask(img=face, coordinate=coordinates[idx]):
                face = self.put_mask(face, coordinates[idx])

            f_[idx] = face
            idx += 1

        return f_
