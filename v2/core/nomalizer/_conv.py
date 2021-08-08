import numpy as np
import cv2

# model
from ._base import BaseNormalizer

# from v2.core.exceptions import *


class GrayScaleConvertor(BaseNormalizer):

    def __init__(self, name=None, *args, **kwargs):
        super(GrayScaleConvertor, self).__init__(name, *args, **kwargs)

    def normalize(self, mat: np.ndarray, channel="full") -> np.ndarray:

        if channel == "full":
            _mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
            _mat = np.dstack([_mat, _mat, _mat])
            return _mat
        elif channel == "one":
            _mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
            return _mat

        else:
            raise ValueError("The value of the channel is not exists")
