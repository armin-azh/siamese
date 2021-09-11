from typing import List, Tuple

# model
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from .base import BaseModel
from v2.core.nomalizer import MaskNormalizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# exceptions
from v2.core.exceptions import *


class MaskClassifierModel(BaseModel):
    def __init__(self, model_path: Path, score_threshold: float, name=None, *args, **kwargs):
        super(MaskClassifierModel, self).__init__(model_path=model_path, name=name, *args, **kwargs)
        self._score_threshold = score_threshold
        self._inputs_name = ["x:0"]
        self._outputs_name = ["Identity:0"]
        self._input_tensor_shape = tf.TensorShape([None, 64, 64, 3])
        self._image_tensor_shape = tf.TensorShape([None, None, 3])
        self._box_tensor_shape = tf.TensorShape([None, 4])
        self._output_tensor_shape = tf.TensorShape([None, 1])
        self._normalizer = MaskNormalizer(name="mask_normalizer_(mobilenet)")

    def predict(self, input_im: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self._image_tensor_shape.assert_is_compatible_with(input_im.shape)
        self._box_tensor_shape.assert_is_compatible_with(boxes.shape)

        if boxes.shape[0] > 0:
            _norm_cropped = self._normalizer.normalize(mat=input_im,
                                                       b_mat=boxes,
                                                       interpolation=cv2.INTER_LINEAR,
                                                       offset_per=0,
                                                       cropping="large")
            self._input_tensor_shape.assert_is_compatible_with(_norm_cropped.shape)
            _norm_cropped = preprocess_input(_norm_cropped)
            return self._model.predict(_norm_cropped)
        else:
            return np.empty((0, 1))

    def __validate_no_mask(self, mat: np.ndarray) -> np.ndarray:
        _ans = np.where(mat[:, :] <= self._score_threshold)
        return _ans[0]

    def __validate_mask(self, mat: np.ndarray) -> np.ndarray:
        _ans = np.where(mat[:, :] > self._score_threshold)
        return _ans[0]

    def validate_mask(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._output_tensor_shape.assert_is_compatible_with(mat.shape)
        if mat.shape[0] > 0:
            return self.__validate_mask(mat), self.__validate_no_mask(mat)
        else:
            return np.empty((0, 1)), np.empty((0, 1))
