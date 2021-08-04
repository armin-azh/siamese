import numpy as np

# base
from .base import BaseDistance

# exceptions
from v2.core.exceptions import InCompatibleDimError


class CosineDistance(BaseDistance):
    def __init__(self, similarity_threshold: float, name=None, *args, **kwargs):
        self._sim_threshold = similarity_threshold
        super(CosineDistance, self).__init__(name, *args, **kwargs)

    def __cosine_similarity_1_k(self, n_obs: np.ndarray, bs_obs: np.ndarray) -> np.ndarray:
        """
        this function calculate embedding cosine distance respect to embeddings
        :param bs_obs:
        :param n_obs:
        :return: vector in shape (n,)
        """
        dot = np.sum(np.multiply(n_obs, bs_obs), axis=1)
        norm = np.linalg.norm(n_obs, axis=1) * np.linalg.norm(bs_obs, axis=1)
        dist = np.arccos(dot / norm) / np.pi
        return dist

    def calculate_distant(self, n_obs: np.ndarray, bs_obs: np.ndarray) -> np.ndarray:
        """
        :param n_obs: matrix in shape (n,m)
        :param bs_obs: matrix in shape (k,m)
        :return: matrix in shape (n,m)
        """

        if len(n_obs.shape) != len(bs_obs.shape):
            raise InCompatibleDimError("The n_obs and bs_obs are not the same")

        if n_obs.shape[1] != bs_obs.shape[1]:
            raise InCompatibleDimError("The n_obs is not compatible with bs_obs")

        dists = list()
        for idx in range(n_obs.shape[0]):
            vec = np.expand_dims(n_obs[idx], axis=0)
            dists.append(self.__cosine_similarity_1_k(vec, bs_obs))
        return np.array(dists)

    def satisfy(self, r_obs: np.ndarray) -> np.ndarray:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs]
