import numpy as np
import cv2
import tensorflow as tf

# model
from ._base import BaseNormalizer

from v2.core.exceptions import *


class FaceNetNormalizer(BaseNormalizer):
    def __init__(self, name=None, *args, **kwargs):
        super(FaceNetNormalizer, self).__init__(name, *args, **kwargs)

    def normalize(self, mat: np.ndarray, **kwargs) -> np.ndarray:
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


class HeadPoseEstimatorNormalizer(BaseNormalizer):
    def __init__(self, name=None, *args, **kwargs):
        self._box_tensor_shape = tf.TensorShape([None, 4])
        super(HeadPoseEstimatorNormalizer, self).__init__(name, *args, **kwargs)

    def normalize(self, mat: np.ndarray, **kwargs) -> np.ndarray:
        """

        :param mat: a gray level matrix in shape (w,h)
        :param kwargs:
        :return:
        """
        offset_per = kwargs["offset_per"]
        cropping = kwargs["cropping"]
        interpolation = kwargs["interpolation"]
        hpe_im_norm = kwargs["hpe_im_norm"]
        b_mat = kwargs.get("b_mat")
        if b_mat is None:
            raise NoPassingArgumentError("bounding boxes had`t passed.")

        self._box_tensor_shape.assert_is_compatible_with(b_mat.shape)
        if b_mat.shape[0] == 0:
            return np.empty((0, 64, 64, 1))

        if len(mat.shape) == 2:
            _ori_height = mat.shape[0]
            _ori_width = mat.shape[1]
            _cropped = []
            for box in b_mat:
                _x_min = box[0]
                _y_min = box[1]
                _x_max = box[2]
                _y_max = box[3]

                if cropping == 'large':
                    if (_x_max - _x_min) > (_y_max - _y_min):
                        _y_min = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                        _y_max = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

                    elif (_y_max - _y_min) > (_x_max - _x_min):
                        _x_min = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                        _x_max = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1
                elif cropping == 'small':

                    if (_x_max - _x_min) > (_y_max - _y_min):
                        _x_min = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                        _x_max = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

                    elif (_y_max - _y_min) > (_x_max - _x_min):
                        _y_min = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                        _y_max = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

                __new_size = _x_max - _x_min

                _x_min = _x_min - int(__new_size * offset_per)
                _y_min = _y_min - int(__new_size * offset_per)

                _x_max = _x_max + int(__new_size * offset_per)
                _y_max = _y_max + int(__new_size * offset_per)

                if _x_min < 0:
                    _x_max = _x_max - _x_min
                    _x_min = 0

                if _x_max >= (_ori_width - 1):
                    _x_min = (_ori_width - 1) - (_x_max - _x_min)
                    _x_max = _ori_width - 1

                if _y_min < 0:
                    _y_max = _y_max - _y_min
                    _y_min = 0

                if _y_max >= (_ori_height - 1):
                    _y_min = (_ori_height - 1) - (_y_max - _y_min)
                    _y_max = _ori_height - 1

                if _x_min >= 0 and _y_min >= 0 and _x_max < _ori_width and _y_max < _ori_height:
                    c_pic = mat[_y_min:_y_max, _x_min:_x_max]
                    if cropping == 'small' or cropping == 'large':
                        c_pic = cv2.resize(c_pic,
                                           (int(64 * (1 + 2 * offset_per)),
                                            int(64 * (1 + 2 * offset_per))),
                                           interpolation=interpolation)
                    else:
                        if c_pic.shape[0] > c_pic.shape[1]:
                            c_pic = cv2.resize(c_pic,
                                               (int((64 * c_pic.shape[0] / c_pic.shape[1]) * (
                                                       1 + 2 * offset_per)),
                                                int(64 * (1 + 2 * offset_per))),
                                               interpolation=interpolation)
                        else:
                            c_pic = cv2.resize(c_pic, (int(64 * (1 + 2 * offset_per)),
                                                       int((64 * c_pic.shape[1] / c_pic.shape[0]) * (
                                                               1 + 2 * offset_per))), interpolation=interpolation)
                    _cropped.append(c_pic)
                else:
                    _cropped.append(np.empty(0))

            _cropped = np.array(_cropped)
            _cropped = np.expand_dims(_cropped, axis=-1)
            _norm_cropped = ((_cropped / 255.) - hpe_im_norm[0]) / hpe_im_norm[1]
            return _norm_cropped

        else:
            raise InCompatibleDimError("Unknown image dimension")
