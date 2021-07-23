import tensorflow as tf
import cv2
import numpy as np
from typing import Tuple


class HPE:
    def __init__(self, img_norm: Tuple[int, int], tilt_norm: Tuple[int, int], pan_norm: Tuple[int, int],
                 rescale: float):
        self._img_norm = img_norm
        self._tilt_norm = tilt_norm
        self._pan_norm = pan_norm
        self._rescale = rescale
        self._input_size = 64

    def normalize_images(self, img: np.ndarray) -> np.ndarray:
        """
        get position of the input images
        :param img: images in shape (m,64,64,1)
        :return:
        """
        img_norm = ((img / 255.) - self._img_norm[0]) / self._img_norm[1]

        return img_norm

    def normalize_pose(self, poses: np.ndarray) -> np.ndarray:
        """
        normalize predicted pose
        :param poses: tensor in shape (m,2)
        :return: normalized poses
        """

        poses[:, 0] = (poses[:, 0] * self._tilt_norm[1] + self._tilt_norm[0]) * self._rescale
        poses[:, 1] = (poses[:, 1] * self._pan_norm[1] + self._pan_norm[0]) * self._rescale

        return poses

    def predict(self, sess: tf.compat.v1.Session, img: np.ndarray) -> np.ndarray:
        """
        predict the tilt and pan
        :param sess: tensorflow session
        :param img: tensor in shape (m,64,64,1)
        :return: tensor in shape (m,2)
        """

        img_norm = self.normalize_images(img)

        pred = sess.run()
