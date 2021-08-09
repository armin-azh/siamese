# from typing import List, Tuple

# model
import numpy as np
import tensorflow as tf

from .base import BaseModel
from v2.core.nomalizer import FaceNetNormalizer

# exceptions
from v2.core.exceptions import *


class FaceNetModel(BaseModel):
    def __init__(self, model_path, name=None, *args, **kwargs):
        super(FaceNetModel, self).__init__(model_path=model_path,
                                           name=name, *args, **kwargs)
        self._inputs_name = ["input:0"]
        self._outputs_name = ["embeddings:0"]
        self._input_tensor_shape = tf.TensorShape([None, 160, 160, 3])
        self._normalizer = FaceNetNormalizer(name="faceNetNormalizer")

    def get_embeddings(self, session: tf.compat.v1.Session, input_im: np.ndarray) -> np.ndarray:
        _in_shape = input_im.shape
        try:
            self._input_tensor_shape.assert_is_compatible_with(_in_shape)
            _fn_input_plc = self._inputs[0][0]
            _fn_embed_plc = self._outputs[0][0]
            _fn_phase_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            _feed = {_fn_phase_plc: False, _fn_input_plc: self._normalizer.normalize(input_im)}
            return session.run(_fn_embed_plc, feed_dict=_feed)

        except ValueError:
            raise InCompatibleDimError("Input shape is not compatible with model shape")
