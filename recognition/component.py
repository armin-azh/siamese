from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import configparser
from numpy import load
from database.component import ImageDatabase
from settings import BASE_DIR
import tensorflow as tf


class Provider:
    def __init__(self):
        # tm = os.path.join(BASE_DIR, db_conf.get("database_path"))
        # if not os.path.exists(tm):
        #     os.makedirs(tm)
        conf = configparser.ConfigParser()
        conf.read(os.path.join(BASE_DIR, "conf.ini"))

        self.db_conf = conf['Gallery']
        tm = "G:\\Documents\\Project\\siamese\\data\\train\\set_2\\train"
        self._db = ImageDatabase(db_path=tm)

    def _update(self):
        if self._db.check() == self._db.MODIFIED:
            self._db.update()

    def load_db_encode(self):
        data_path = self.db_conf.get("npy_dir")
        data_path = os.path.join(BASE_DIR, data_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            return None
        path_list = os.listdir(data_path)

        if not path_list:
            return None

        return load(str(os.path.join(data_path, path_list[-1])))

    def encode_images(self):

        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                pass


if __name__ == "__main__":
    obj = Provider()
