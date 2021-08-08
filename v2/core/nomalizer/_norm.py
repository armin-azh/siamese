import numpy as np

# model
from ._base import BaseNormalizer

from v2.core.exceptions import *


class FaceNetNormalizer(BaseNormalizer):
    def __init__(self, name=None, *args, **kwargs):
        super(FaceNetNormalizer, self).__init__(name, *args, **kwargs)

    def normalize(self, mat: np.ndarray) -> np.ndarray:
        """
        :param mat: tensor in shape (n,w,h,c)
        :return: tensor in shape (n,w,h,c)
        """
        _shape = mat.shape
        if len(_shape) != 4:
            raise InCompatibleDimError("dimension of mat tensor is not compatible")

        if _shape[0] > 0:
            _size = mat[0].size
            _mat = mat.reshape((_shape[0], -1))
            means = np.expand_dims(np.mean(_mat, axis=1), axis=-1)
            stds = np.expand_dims(np.std(_mat, axis=1), axis=-1)
            std_adj = np.maximum(stds, 1.0 / np.sqrt(_size))
            _mat = np.multiply(np.subtract(_mat, means), (1 / std_adj))
            return _mat.reshape(_shape)
        else:
            return np.empty_like(mat)

