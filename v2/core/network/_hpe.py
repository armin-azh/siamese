from typing import List, Tuple

# model
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from .base import BaseModel
from v2.core.nomalizer import HpeNormalizer

# exceptions
from v2.core.exceptions import *


class HeadPoseEstimatorModel(BaseModel):
    def __init__(self, model_path: Path, img_norm: Tuple[float, float], tilt_norm: Tuple[float, float],
                 pan_norm: Tuple[float, float],
                 rescale: float, conf: Tuple[float, float, float, float], name=None, *args, **kwargs):
        super(HeadPoseEstimatorModel, self).__init__(model_path=model_path,
                                                     name=name,
                                                     *args,
                                                     **kwargs)
        self._inputs_name = ["x:0"]
        self._outputs_name = ["Identity:0"]
        self._img_norm = img_norm
        self._tilt_norm = tilt_norm
        self._pan_norm = pan_norm
        self._rescale = rescale
        self._input_size = 64
        self._tilt_up = conf[2]
        self._tilt_down = conf[3]
        self._pan_right = conf[1]
        self._pan_left = conf[0]
        self._input_tensor_shape = tf.TensorShape([None, 64, 64, 1])
        self._bounding_box_tensor_shape = tf.TensorShape([None, 4])
        self._poses_tensor_shape = tf.TensorShape([None, 2])
        self._normalizer = HpeNormalizer(name="pose-estimator")

    def estimate_poses(self, session: tf.compat.v1.Session, input_im: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self._bounding_box_tensor_shape.assert_is_compatible_with(boxes.shape)
        _in_shape = input_im.shape
        try:
            self._input_tensor_shape.assert_is_compatible_with(_in_shape)
            _norm_cropped = self._normalizer.normalize(mat=input_im,
                                                       b_mat=boxes,
                                                       interpolation=cv2.INTER_LINEAR,
                                                       hpe_im_norm=self._img_norm,
                                                       offset_per=0,
                                                       cropping="large")

            self._input_tensor_shape.assert_is_compatible_with(_norm_cropped)
            _fn_input_plc = self._inputs[0][0]
            _fn_embed_plc = self._outputs[0][0]
            _feed = {_fn_input_plc: self._normalizer.normalize(input_im)}
            _poses = session.run(_fn_embed_plc, feed_dict=_feed)
            return self._normalizer.normalize_output(_poses, self._tilt_norm, self._pan_norm, self._rescale)

        except ValueError:
            raise InCompatibleDimError("Input shape is not compatible with model shape")

    def __validate_angle(self, mat: np.ndarray) -> np.ndarray:
        tilt_log = np.logical_and(mat[:, 0] < self._tilt_up, mat[:, 0] > self._tilt_down)
        pan_log = np.logical_and(mat[:, 1] < self._pan_left, mat[:, 1] > self._pan_right)
        to_log = np.logical_and(tilt_log, pan_log)
        ans = np.where(to_log)
        return ans[0]

    def __validate_angle_complement(self, mat: np.ndarray) -> np.ndarray:
        tilt_log = np.logical_or(mat[:, 0] > self._tilt_up, mat[:, 0] < self._tilt_down)
        pan_log = np.logical_or(mat[:, 1] > self._pan_left, mat[:, 1] < self._pan_right)
        to_log = np.logical_or(tilt_log, pan_log)
        ans = np.where(to_log)
        return ans[0]

    def validate_angle(self, mat: np.ndarray):
        self._poses_tensor_shape.assert_is_compatible_with(mat.shape)
        if mat.shape[0] > 0:
            return self.__validate_angle(mat), self.__validate_angle_complement(mat)
        else:
            return np.empty((0, 2)), np.empty((0, 2))
