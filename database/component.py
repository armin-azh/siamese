from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import json
import pathlib
import configparser
from typing import Tuple
from settings import BASE_DIR
from PIL import Image as PilImage
import numpy as np
from itertools import chain
from sklearn import preprocessing
from utils import extract_filename

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

    def read_image_file(self, resize: bool = False):
        """
        read image file from file
        :return:
        """
        im = PilImage.open(self._im_path)
        return self._preprocessing(im, resize)

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
        base_dir, file_path = extract_filename(self._im_path)
        return os.path.join(base_dir, file_path + '.json')

    def _gen_csv_filename(self) -> str:
        """
        generate csv filename base on image folder path
        :return: str
        """
        base_dir, file_path = extract_filename(self._im_path)
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


class Identity:
    __slots__ = ['_name', '_images']
    identities_name = []

    def __init__(self, name: str):
        self._name = name
        self._images = []

    def __del__(self):
        Identity.identities_name.remove(self._name)

    @classmethod
    def create(cls, name):
        name, status = cls.add_new_id(name)
        if not status:
            return Identity(name)
        else:
            return None

    @classmethod
    def add_new_id(cls, name: str) -> tuple:
        status = True if name in cls.identities_name else False
        cls.identities_name.append(name)
        return name, status

    @classmethod
    def reset(cls):
        """
        this method reset class state
        :return: None
        """
        cls.identities_name = []

    def add_image(self, im_path):
        """
        add new image to the identity
        :param im_path:
        :return: None
        """
        self._images.append(Image(im_path=im_path))

    def get_images_path(self):
        """
        generator for return path
        :return:
        """
        for im_path in self._images:
            yield im_path

    @property
    def images_len(self):
        return len(self._images)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name, status = Identity.add_new_id(name)
        if not status:
            self._name = name


class ImageDatabase:
    __slots__ = ['_identities', '_db_path', '_db_json_path']
    COMMITTED = 'committed'
    MODIFIED = "modified"

    def __init__(self, db_path):
        self._db_path = db_path
        self._identities = list()
        self._db_json_path = self._gen_json_filename()
        if not self._db_json_path.exists():
            self.initiate_conf_file()
        self.update()

    def get_identity_image_paths(self):
        return {iden.name: iden.get_images_path() for iden in self._identities}

    def add_identity(self, iden: Identity):
        self._identities.append(iden)

    def parse(self):
        """
        parse Gallery set
        :return:
        """
        base = pathlib.Path(self._db_path)
        ids_dict = dict()
        for ch in base.glob('**'):
            if ch.stem != base.stem:
                images = list()
                for p in ch.glob('**/*.*'):
                    images.append(str(p))
                ids_dict[ch.stem] = images
        return ids_dict

    def _gen_json_filename(self):
        base = pathlib.Path(self._db_path)
        return base.joinpath('conf.json')

    def initiate_conf_file(self):
        temp = dict()
        temp['data'] = dict()
        temp['commit'] = False
        self._write_json_file(temp)

    def load_json_file(self):
        with open(self._db_json_path, 'r') as infile:
            conf_file = json.load(infile)
        return conf_file

    def _write_json_file(self, conf_dic: dict):
        with open(self._db_json_path, 'w') as outfile:
            json.dump(conf_dic, outfile)

    def check(self):
        conf_file = self.load_json_file()
        if conf_file.get('commit'):
            return self.COMMITTED
        else:
            return self.MODIFIED

    def update(self):
        if self.check() == self.MODIFIED:
            conf_data = dict()
            conf_data['data'] = self.parse()
            conf_data['commit'] = True
            self._write_json_file(conf_data)
            self._in_memory_load()
        else:
            self._in_memory_load()

    def _in_memory_load(self):
        conf_data = self.load_json_file()
        data = conf_data.get('data')

        for key, value in data.items():
            tm_id = Identity.create(key)
            for im_path in chain(value):
                tm_id.add_image(im_path)
            self._identities.append(tm_id)

    @staticmethod
    def split_data_and_label(dataset_dictionary):
        """
        this method split the dataset to its data and label
        :param dataset_dictionary:
        :return: data list and labels
        """

        images_list = []
        labels = []
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(list(dataset_dictionary.keys()))
        for idx, (identity, images) in enumerate(dataset_dictionary.items()):
            images_list += list(images)
            labels += [idx] * len(list(images))

        return images_list, labels, label_encoder


if __name__ == "__main__":
    obj = ImageDatabase('G:\\Documents\\Project\\siamese\\data\\train\\set_2\\train')
    ImageDatabase.split_data_and_label(obj.get_identity_image_paths())
