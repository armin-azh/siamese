import os
import numpy as np
from numpy import genfromtxt
from PIL import Image
import PIL

DEFAULT_IMAGE_SIZE = (96, 96)

WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
    'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'dense_layer'
]

conv_shape = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1],
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1, ],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1],
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1],
    'inception_5b_1x1_conv': [256, 736, 1, 1],
}


class Loader(object):
    def __init__(self, weight_path=None):
        self._weight_path = weight_path if weight_path is not None else "./weights"
        self._weight_name = WEIGHTS

    def _load_weight(self):
        filenames = filter(lambda f: not f.startswith('.'), os.listdir(self._weight_path))
        paths = dict()
        weights = dict()

        for f_name in filenames:
            paths[f_name.replace('.csv', '')] = os.path.join(self._weight_path, f_name)

        for name in WEIGHTS:
            if 'conv' in name:
                conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
                conv_w = np.reshape(conv_w, conv_shape[name])
                conv_w = np.transpose(conv_w, (2, 3, 1, 0))
                conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
                weights[name] = [conv_w, conv_b]
            elif 'bn' in name:
                bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
                bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
                bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
                bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
                weights[name] = [bn_w, bn_b, bn_m, bn_v]
            elif 'dense' in name:
                dense_w = genfromtxt(self._weight_path + '/dense_w.csv', delimiter=',', dtype=None)
                dense_w = np.reshape(dense_w, (128, 736))
                dense_w = np.transpose(dense_w, (1, 0))
                dense_b = genfromtxt(self._weight_path + '/dense_b.csv', delimiter=',', dtype=None)
                weights[name] = [dense_w, dense_b]

        return weights

    def load_weights(self, model_obj):
        weight_dict = self._load_weight()
        weight_name = self.get_weight_name()

        for name in weight_name:
            model_obj.get_layer(name).set_weights(weight_dict[name])

    def get_weight_name(self):
        return self._weight_name


class ImageDatabase(object):
    def __init__(self, database_path):
        if os.listdir(database_path):
            self._db_path = database_path
        else:
            raise ValueError

        self._image_names = self._get_image_names()
        self._update_images()
        self._image_db = self._get_image_name_dictionary()

    def _get_image_names(self):
        return os.listdir(self._db_path)

    def _get_image_name_dictionary(self):
        tp = dict()
        for name in self._image_names:
            ims_paths = os.path.join(self._db_path, name)
            for image_name in os.listdir(ims_paths):
                im_path = os.path.join(ims_paths, image_name)
                image = Image.open(im_path)
                tp[name] = np.array(image)
        return tp

    def get_images(self):
        return self._image_db

    def _update_images(self):
        for name in self._image_names:
            ims_paths = os.path.join(self._db_path, name)
            for image_name in os.listdir(ims_paths):
                im_path = os.path.join(ims_paths, image_name)
                image = Image.open(im_path)
                if image.size != DEFAULT_IMAGE_SIZE:
                    image = image.resize(DEFAULT_IMAGE_SIZE)
                    image.save(im_path)

    @staticmethod
    def default_size(image_pil):
        if isinstance(image_pil, PIL.JpegImagePlugin.JpegImageFile):
            return image_pil.resize(DEFAULT_IMAGE_SIZE)
        else:
            raise ValueError

