import numpy as np
from typing import Tuple


class BaseDistance:
    def __init__(self, name=None, *args, **kwargs):
        self._name = self.__class__.__name__ if name is None else name
        super(BaseDistance, self).__init__(*args, **kwargs)

    def calculate_distant(self, n_obs: np.ndarray, bs_obs: np.ndarray) -> np.ndarray:
        """
        calculate distant between new observation and base observation
        :param n_obs: new observation
        :param bs_obs: base observation
        :return:
        """
        raise NotImplementedError

    def satisfy(self, r_obs: np.ndarray) -> np.ndarray:
        """
        this method extract the index there we want
        :param r_obs: calculate_distant result
        :return: indexes
        """
        raise NotImplementedError

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        this method validate r_obs values base on a threshold
        :param r_obs:
        :return: validate and not validate indexes
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
