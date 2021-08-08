from typing import List, Tuple

# model
import numpy as np

from .base import BaseModel
from .mtcnn import detect_face

# exceptions
from v2.core.exceptions import *


class FaceDetector(BaseModel):
    def __init__(self, stages_threshold, scale_factor: float, min_face: int, name=None,
                 *args, **kwargs):
        self._stages_threshold = stages_threshold
        self._scale_factor = scale_factor
        self._min_face = min_face
        self._type = "Unknown"
        self._p_net_fn, self._r_net_fn, self._o_net_fn = None, None, None
        super(FaceDetector, self).__init__(model_path=None, name=name, *args, **kwargs)

    @property
    def inputs(self):
        raise DisableMethodWarning("inputs method has disabled by child class")

    @property
    def outputs(self):
        raise DisableMethodWarning("outputs method has disabled by child")

    def set_inputs_name(self, names: List[str]) -> None:
        raise DisableMethodWarning("set_inputs_name method has disabled by child class")

    def set_outputs_name(self, names: List[str]) -> None:
        raise DisableMethodWarning("set_outputs_name method has disabled by child class")

    def load_model(self, **kwargs):
        _sess = kwargs.get("session")
        if _sess is None:
            raise SessionIsNotSetError("you should set session on load_model")

        self._p_net_fn, self._r_net_fn, self._o_net_fn = detect_face.create_mtcnn(sess=_sess)

    def extract(self, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return detect_face.detect_face(im,
                                       minsize=self._min_face,
                                       pnet=self._p_net_fn,
                                       rnet=self._r_net_fn,
                                       onet=self._o_net_fn,
                                       threshold=self._stages_threshold,
                                       factor=self._scale_factor
                                       )
