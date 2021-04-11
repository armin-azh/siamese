from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
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
        return np.array(self._preprocessing(im))

    def _preprocessing(self, im):
        """
        :param im: pillow image datatype
        :return:
        """
        im = im.resize(self.__size)
        return im

    def _generate_csv_file_name(self):
        pass
