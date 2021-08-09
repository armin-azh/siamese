import numpy as np


class BaseNormalizer:
    def __init__(self, name=None, *args, **kwargs):
        self._name = self.__class__.__name__ if name is None else name

        super(BaseNormalizer, self).__init__(*args, **kwargs)

    def normalize(self, mat: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
