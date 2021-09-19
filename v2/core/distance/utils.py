import numpy as np


def calc_cov_g_inverse(mat: np.ndarray):
    """
    compute general inverse
    :param mat: is a mxn matrix
    :return: general inverse matrix
    """
    return np.linalg.pinv(np.cov(mat.T))
