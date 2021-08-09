from typing import List, Tuple

# model
import numpy as np
import tensorflow as tf
from pathlib import Path

from .base import BaseModel
from v2.core.nomalizer import FaceNetNormalizer

# exceptions
from v2.core.exceptions import *


class HeadPoseEstimatorModel(BaseModel):
    def __init__(self, model_path:Path, img_norm: Tuple[float, float], tilt_norm: Tuple[float, float],
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

    def estimate_poses(self, session: tf.compat.v1.Session, input_im: np.ndarray) -> np.ndarray:
        _in_shape = input_im.shape
        try:
            self._input_tensor_shape.assert_is_compatible_with(_in_shape)
            _fn_input_plc = self._inputs[0][0]
            _fn_embed_plc = self._outputs[0][0]
            _feed = {_fn_input_plc: self._normalizer.normalize(input_im)}
            return session.run(_fn_embed_plc, feed_dict=_feed)

        except ValueError:
            raise InCompatibleDimError("Input shape is not compatible with model shape")
