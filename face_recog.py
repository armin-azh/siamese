from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K_1
import numpy as np
from .utils import ImageDatabase
from .model import FaceNet

import os
import sys


class FaceRecognition(object):
    def __init__(self, model_path, image_db_path=None, distance_threshold=.7):
        self._threshold = distance_threshold
        if os.path.isfile(model_path):
            self._model_path = model_path
        else:
            raise ValueError

        self._model = load_model(model_path, custom_objects={'triplet_loss': FaceNet.triplet_loss})
        self._im_db_path = './database' if image_db_path is None else image_db_path
        self._im_db = ImageDatabase(database_path=self._im_db_path)
        self._encoded_im_db = self._encode_database()

    def verify(self, image):
        en_test_im = self.encode_image(image)
        min_distance = sys.maxsize

        identified_name = None

        for name, en_im in self._encoded_im_db.items():
            ds = np.linalg.norm(en_test_im - en_im)

            if ds < min_distance:
                min_distance = ds
                identified_name = name

        if min_distance > self._threshold:
            return None
        else:
            return identified_name

    def _encode_database(self):
        encoded_im_db = dict()
        for name, im in self._im_db.get_images().items():
            encode = self.encode_image(im)
            encoded_im_db[name] = encode
        return encoded_im_db

    def encode_image(self, image):
        image = image[..., ::-1]
        image = np.around(np.transpose(image, (2, 0, 1)) / 255., decimals=12)
        input_im = np.array([image])
        return self._model.predict(input_im)

