from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K_1
import numpy as np
from .image_database import ImageDatabase, Image
from .model import FaceNet

import os
import sys


class FaceRecognition(object):
    """
    this class implement the face recognition model
    """

    def __init__(self, model_path, image_db_path=None, distance_threshold=1.0):
        self._threshold = distance_threshold
        if os.path.isfile(model_path):
            self._model_path = model_path
        else:
            raise ValueError

        self._model = load_model(model_path, custom_objects={'triplet_loss': FaceNet.triplet_loss})
        self._im_db_path = './database' if image_db_path is None else image_db_path
        self._im_db = ImageDatabase(db_path=self._im_db_path)

    @property
    def threshold(self):
        """
        get min distance threshold
        :return:
        """
        return self._threshold

    @threshold.setter
    def threshold(self, n_thresh):
        """
        set min distance threshold
        :param n_thresh: float between 0 and 1
        :return:
        """
        self._threshold = n_thresh

    def inception_verify(self, test_images, n_images=None):
        """

        :param n_images: int fetch number of images for each identities
        :param test_images: dictionary of test images
        :return: dictionary
        """
        res = dict()
        encoded_identities = self._im_db.get_encoded_identities_images(model=self._model, n_images=n_images)
        for t_name, t_tensor in test_images.items():
            en_t_tensor = Image.encode(model=self._model, image=t_tensor)

            distances = list()
            for identity, en_id_tensor in encoded_identities.items():
                distances.append((identity, FaceRecognition.calc_distance(en_t_tensor, en_id_tensor)))

            res[t_name] = distances

        return res

    @staticmethod
    def calc_distance(tensor_1, tensor_2):
        """
        compute l2 distance
        :param tensor_1:
        :param tensor_2:
        :return:
        """
        return np.linalg.norm(tensor_1 - tensor_2)

    def verify(self, image, n_images=None):
        """
        this method recognize a person in identities database
        :param n_images: int fetch number of images for each identities
        :param image: numpy tensor
        :return: tuple(identity,min_distance,status)
        """
        en_t_tensor = Image.encode(model=self._model, image=image)

        min_distance = sys.maxsize

        identified_name = None

        en_identities = self._im_db.get_encoded_identities_images(model=self._model, n_images=n_images)
        for identity, en_tensors in en_identities.items():
            distances = FaceRecognition.calc_distance(en_t_tensor, en_tensors)

            if np.min(distances) < min_distance:
                min_distance = np.min(distances)
                identified_name = identity

        if min_distance > self._threshold:
            return min_distance, identified_name, "Not in database, Access Denied!"
        else:
            return min_distance, identified_name, "In database, Access Granted!"

