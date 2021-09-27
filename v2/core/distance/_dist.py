import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import mahalanobis
from typing import Tuple

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

    def satisfy(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs], min_obs

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _valid_ans = np.where(r_obs <= self._sim_threshold)
        _invalid_ans = np.where(r_obs > self._sim_threshold)
        return _valid_ans[0], _invalid_ans[0]


class SklearnCosineDistance(BaseDistance):

    def __init__(self, similarity_threshold: float, name=None, *args, **kwargs):
        self._sim_threshold = similarity_threshold
        super(SklearnCosineDistance, self).__init__(name, *args, **kwargs)

    def satisfy(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs], min_obs

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

        return cosine_distances(n_obs, bs_obs)

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _valid_ans = np.where(r_obs <= self._sim_threshold)
        _invalid_ans = np.where(r_obs > self._sim_threshold)
        return _valid_ans[0], _invalid_ans[0]


class EuclideanDistance(BaseDistance):
    def __init__(self, similarity_threshold: float, name=None, *args, **kwargs):
        self._sim_threshold = similarity_threshold
        super(EuclideanDistance, self).__init__(name, *args, **kwargs)

    def satisfy(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs], min_obs

    def __euclidean_similarity_1_k(self, n_obs: np.ndarray, bs_obs: np.ndarray) -> np.ndarray:
        """
        this function calculate embedding euclidean distance respect to embeddings
        :param bs_obs:
        :param n_obs:
        :return: vector in shape (n,)
        """
        return np.linalg.norm(n_obs - bs_obs, axis=1)

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
            dists.append(self.__euclidean_similarity_1_k(vec, bs_obs))
        return np.array(dists)

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _valid_ans = np.where(r_obs <= self._sim_threshold)
        _invalid_ans = np.where(r_obs > self._sim_threshold)
        return _valid_ans[0], _invalid_ans[0]


class SklearnEuclideanDistance(BaseDistance):

    def __init__(self, similarity_threshold: float, name=None, *args, **kwargs):
        self._sim_threshold = similarity_threshold
        super(SklearnEuclideanDistance, self).__init__(name, *args, **kwargs)

    def satisfy(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs], min_obs

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

        return euclidean_distances(n_obs, bs_obs)

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _valid_ans = np.where(r_obs <= self._sim_threshold)
        _invalid_ans = np.where(r_obs > self._sim_threshold)
        return _valid_ans[0], _invalid_ans[0]


class SiPyMahalanobisDistance512(BaseDistance):
    def __init__(self, similarity_threshold: float, name=None, *args, **kwargs):
        self._sim_threshold = similarity_threshold
        super(SiPyMahalanobisDistance512, self).__init__(name, *args, **kwargs)

    def calculate_distant(self, n_obs: np.ndarray, bs_obs: np.ndarray, inv_mat: np.ndarray) -> np.ndarray:
        """
        :param n_obs: matrix in shape (n,512)
        :param bs_obs: matrix in shape (k,512)
        :param inv_mat: tensor in shape (k, 512, 512)
        :return: matrix in shape (n,k)
        """

        if len(n_obs.shape) != len(bs_obs.shape):
            raise InCompatibleDimError("The n_obs and bs_obs are not the same")

        if n_obs.shape[1] != bs_obs.shape[1]:
            raise InCompatibleDimError("The n_obs is not compatible with bs_obs")

        if bs_obs.shape[0] != inv_mat.shape[0]:
            raise InCompatibleDimError("The bs_obs is not compatible with inv_mat")

        _dists = np.empty((n_obs.shape[0], bs_obs.shape[0]), dtype=np.float32)
        for _idx, n_sample in enumerate(n_obs):
            for _idy in range(bs_obs.shape[0]):
                _dists[_idx, _idy] = mahalanobis(n_sample, bs_obs[_idy], inv_mat[_idy].reshape((512, 512)))

        return _dists

    def satisfy(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param r_obs: matrix in shape (n,m)
        :return:
        """
        if len(r_obs.shape) != 2:
            raise InCompatibleDimError("The r_obs is not 2 dimension")
        min_obs = np.argmin(r_obs, axis=1)
        return r_obs[np.arange(len(min_obs)), min_obs], min_obs

    def validate(self, r_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _valid_ans = np.where(r_obs <= self._sim_threshold)
        _invalid_ans = np.where(r_obs > self._sim_threshold)
        return _valid_ans[0], _invalid_ans[0]
