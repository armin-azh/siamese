import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
import cv2
import numpy as np
from typing import Tuple


class HPE:
    INPUT_SHAPE = None

    def __init__(self, img_norm: Tuple[float, float], tilt_norm: Tuple[float, float], pan_norm: Tuple[float, float],
                 rescale: float):
        self._img_norm = img_norm
        self._tilt_norm = tilt_norm
        self._pan_norm = pan_norm
        self._rescale = rescale
        self._input_size = 64
        self.INPUT_SHAPE = self._input_size

    def reshape_and_convert(self, img: np.ndarray) -> np.ndarray:
        """

        :param img: tensor in shape (width,height,channel)
        :return: tensor in shape (m,64,64,1)
        """

        pro_img = np.reshape(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self._input_size, self._input_size)),
                             (self._input_size, self._input_size, 1))

        return pro_img

    def reshape_and_convert_bulk(self, img: np.ndarray) -> np.ndarray:
        """

        :param img: tensor in shape (m,width,height,channel)
        :return: tensor in shape (m,64,64,1)
        """
        pro_img = np.empty((img.shape[0], self._input_size, self._input_size, 1))

        for idx, tm_tm in enumerate(img):
            pro_img[idx, :, :, :] = np.reshape(cv2.resize(cv2.cvtColor(img[idx, :, :, :], cv2.COLOR_BGR2GRAY),
                                                          (self._input_size, self._input_size)),
                                               (self._input_size, self._input_size, 1))

        return pro_img

    def normalize_images(self, img: np.ndarray) -> np.ndarray:
        """
        get position of the input images
        :param img: images in shape (m,64,64,1)
        :return: tensor ins shape (m,64,64,1)
        """

        img_norm = ((img / 255.) - self._img_norm[0]) / self._img_norm[1]

        return img_norm

    def normalize_pose(self, poses: np.ndarray) -> np.ndarray:
        """
        normalize predicted pose
        :param poses: tensor in shape (m,2)
        :return: normalized poses
        """

        poses[:, 0] = (poses[:, 0] * self._tilt_norm[1] + self._tilt_norm[0]) * self._rescale
        poses[:, 1] = (poses[:, 1] * self._pan_norm[1] + self._pan_norm[0]) * self._rescale

        return poses

    def predict(self, sess: tf.compat.v1.Session, img: np.ndarray, input_tensor: Tensor,
                output_tensor: Tensor) -> np.ndarray:
        """
        predict the tilt and pan
        :param input_tensor:
        :param output_tensor:
        :param sess: tensorflow session
        :param img: tensor in shape (m,64,64,1)
        :return: tensor in shape (m,2)
        """
        if img.shape[0] > 0:

            img_norm = self.normalize_images(img)

            feed_dic = {input_tensor: img_norm}

            poses = sess.run(output_tensor, feed_dict=feed_dic)

            return self.normalize_pose(poses)

        else:
            return np.empty((0, 2))

    def get_cropped_pics(self, img: np.ndarray, boxes: np.ndarray, offset_perc: int, cropping: str = '',
                         interpolation=cv2.INTER_LINEAR) -> np.ndarray:

        ori_height = img.shape[0]
        ori_width = img.shape[1]

        pics = []

        for box in boxes:

            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            if cropping == 'large':

                if (xmax - xmin) > (ymax - ymin):
                    ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                    ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

                elif (ymax - ymin) > (xmax - xmin):
                    xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                    xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

            elif cropping == 'small':

                if (xmax - xmin) > (ymax - ymin):
                    xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                    xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

                elif (ymax - ymin) > (xmax - xmin):
                    ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                    ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

            new_size = xmax - xmin

            xmin = xmin - int(new_size * offset_perc)
            ymin = ymin - int(new_size * offset_perc)

            xmax = xmax + int(new_size * offset_perc)
            ymax = ymax + int(new_size * offset_perc)

            if xmin < 0:
                xmax = xmax - xmin
                xmin = 0

            if xmax >= (ori_width - 1):
                xmin = (ori_width - 1) - (xmax - xmin)
                xmax = ori_width - 1

            if ymin < 0:
                ymax = ymax - ymin
                ymin = 0

            if ymax >= (ori_height - 1):
                ymin = (ori_height - 1) - (ymax - ymin)
                ymax = ori_height - 1

            if xmin >= 0 and ymin >= 0 and xmax < ori_width and ymax < ori_height:
                c_pic = img[ymin:ymax, xmin:xmax]
                if cropping == 'small' or cropping == 'large':
                    c_pic = cv2.resize(c_pic,
                                       (int(self._input_size * (1 + 2 * offset_perc)),
                                        int(self._input_size * (1 + 2 * offset_perc))),
                                       interpolation=interpolation)
                else:
                    if c_pic.shape[0] > c_pic.shape[1]:
                        c_pic = cv2.resize(c_pic,
                                           (int((self._input_size * c_pic.shape[0] / c_pic.shape[1]) * (
                                                   1 + 2 * offset_perc)),
                                            int(self._input_size * (1 + 2 * offset_perc))), interpolation=interpolation)
                    else:
                        c_pic = cv2.resize(c_pic, (int(self._input_size * (1 + 2 * offset_perc)),
                                                   int((self._input_size * c_pic.shape[1] / c_pic.shape[0]) * (
                                                           1 + 2 * offset_perc))), interpolation=interpolation)

                pics.append(c_pic)
            else:
                pics.append(np.empty(0))

        pics = np.array(pics)
        pics = np.expand_dims(pics, axis=-1)
        return pics
