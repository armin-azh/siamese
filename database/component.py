from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import utils
import configparser
from typing import Tuple
from settings import BASE_DIR
from PIL import Image as PilImage
import numpy as np

DT_SIZE = Tuple[int, int]
conf = configparser.ConfigParser()
conf.read(os.path.join(BASE_DIR, "conf.ini"))
image_conf = conf['Image']


class Image:
    """
        this class is for store image paths
    """

    __size = (int(image_conf.get('width')), int(image_conf.get('height')))

    def __init__(self, im_path: str):
        self._im_path = im_path
        self._im_csv_path = self._gen_csv_filename()
        self._im_json_path = self._gen_json_filename()

    def __repr__(self):
        return f"Image: {self._im_path}"

    @classmethod
    def get_size(cls) -> DT_SIZE:
        return cls.__size

    def read_image_file(self):
        """
        read image file from file
        :return:
        """
        im = PilImage.open(self._im_path)
        return self._preprocessing(im)

    def _preprocessing(self, im, resize: bool = False):
        """
        :param im: pillow image datatype
        :return:
        """
        if resize:
            im = im.resize(self.__size)
        im = np.array(im)
        mean = np.mean(im)
        std = np.std(im)
        std_adj = np.maximum(std, 1.0 / np.sqrt(im.size))
        im = np.multiply(np.subtract(im, mean), 1 / std_adj)
        return im

    def _gen_json_filename(self) -> str:
        """
        generate jason file name base on image folder path
        :return: str
        """
        base_dir, file_path = utils.extract_filename(self._im_path)
        return os.path.join(base_dir, file_path + '.json')

    def _gen_csv_filename(self) -> str:
        """
        generate csv filename base on image folder path
        :return: str
        """
        base_dir, file_path = utils.extract_filename(self._im_path)
        return os.path.join(base_dir, file_path + '.csv')

    @property
    def image_path(self):
        return self._im_path

    @property
    def image_csv_path(self):
        return self._im_csv_path

    @property
    def image_json_path(self):
        return self._im_json_path
