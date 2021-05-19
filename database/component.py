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
from .utils import extract_filename, tabulate_print
from recognition.utils import create_random_name
from .npy_builder import builder
from PIL import ImageOps

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

    def read_image_file(self, resize: bool = False, grayscale=True):
        """
        read image file from file
        :return:
        """
        im = PilImage.open(self._im_path)
        return self._preprocessing(im, resize, grayscale)

    def _preprocessing(self, im, resize: bool = False, grayscale=True):
        """
        :param im: pillow image datatype
        :return:
        """
        if resize:
            im = im.resize(self.__size)

        if grayscale:
            im = ImageOps.grayscale(im)
            im = np.array(im)
            im = np.dstack([im, im, im])

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
        for im in chain(im_path):
            self._images.append(Image(im_path=im))

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

    def build_npy(self):
        base = pathlib.Path(self._db_path)
        ids = []
        for ch in base.glob('**'):
            ids_dict = dict()
            if ch.stem != base.stem:
                ids_dict['npy'] = os.path.join(str(ch), ch.stem + '.npy')
                ids_dict['name'] = ch.stem
                images = []
                for p in ch.glob('**/*.jpg'):
                    images.append(Image(im_path=str(p)))
                ids_dict['images'] = images
                ids.append(ids_dict)

        builder(ids)
        self.commit()
        print("$ embedding matrices had been created.")

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
                nps = list()
                for p in ch.glob('**/*.jpg'):
                    images.append(str(p))
                for p in ch.glob('**/*.npy'):
                    nps.append(str(p))
                ids_dict[ch.stem] = (images, nps)

        return ids_dict

    def _parse_npy(self):
        """
        extract npy file from folders
        :return:
        """
        base = pathlib.Path(self._db_path)
        ids_dict = dict()
        for ch in base.glob('**'):
            if ch.stem != base.stem:
                ids_dict[ch.stem] = str(list(ch.glob('**/*.npy'))[0])
        return ids_dict

    def _load_npy(self):
        ids = self._parse_npy()
        for name, npy_path in ids.items():
            yield name, np.load(npy_path)

    def bulk_embeddings(self):
        embeds = []
        labels = []
        for a in self._load_npy():
            embeds.append(a[1])
            labels += [a[0]] * a[1].shape[0]
        return np.concatenate(embeds, axis=0), labels

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
            if tm_id is not None:
                for im_path in chain(value):
                    tm_id.add_image(im_path)
                self._identities.append(tm_id)

    def is_db_stable(self):
        status = True
        has_min_person = False
        for key, value in self.parse().items():
            has_min_person = True
            if not value[1]:
                status = False
                break
        return status and has_min_person

    def save_clusters(self, clusters, faces, cluster_name):
        """
        save cluster images
        :param clusters:
        :param faces:
        :param cluster_name:
        :return:
        """
        base_path = os.path.join(BASE_DIR, self._db_path, cluster_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for key, value in clusters.items():
            idx = np.random.choice(value)
            post = create_random_name(max_length=6)
            save_path = os.path.join(base_path, f"image_{key}_{idx}_{post}.jpg")
            im_ = faces[idx]
            im_ = im_.astype(np.uint8)
            im_ = PilImage.fromarray(im_)
            im_.save(save_path)
        self.modify()
        print(f"$ Images saved at {base_path}")

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

    @staticmethod
    def check_name_exists(name) -> bool:
        """
        check a new identity name
        :param name:
        :return:
        """
        return True if name in Identity.identities_name else False

    def modify(self):
        data = self.load_json_file()
        data['commit'] = False
        self._write_json_file(data)

    def commit(self):
        data = self.load_json_file()
        data['commit'] = True
        self._write_json_file(data)


def inference_db(args):
    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    gallery_conf = conf['Gallery']

    db = ImageDatabase(db_path=gallery_conf.get("database_path"))
    if args.db_check:
        print(f"$ Database had been {db.check()}")

    elif args.db_inspect:
        tabulate_print(db.parse())

    elif args.db_build_npy:
        db.build_npy()


def parse_test_dir(dir_path):
    """
    parse test directory
    :param dir_path: str
    :return:
    """
    base = pathlib.Path(dir_path)
    labels = []
    images = []
    ids = []
    for ch in base.glob("**"):
        if base.stem != ch.stem:
            ims = list()
            for im in ch.glob("**/*.jpg"):
                ims.append(Image(im_path=str(im)))
                # ims.append(im)
            images += ims
            labels += [ch.stem] * len(ims)
            ids.append(ch.stem)
    return images, labels, ids
