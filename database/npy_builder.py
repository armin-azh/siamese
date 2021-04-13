from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import configparser
from settings import BASE_DIR
from itertools import chain
from .utils import load_images
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from recognition.utils import load_model


def builder(ids):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            print(f"$ Initializing computation graph with {model_conf.get('facenet')} pretrained model.")
            load_model(os.path.join(BASE_DIR, model_conf.get('facenet')))
            print("$ Model has been loaded.")

            input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            for a in chain(ids):
                images = load_images(a["images"])
                images = np.array(list(images))
                feed_dic = {phase_train: False, input_plc: images}
                embedded_array = sess.run(embeddings, feed_dic)
                np.save(a['npy'], embedded_array)
                print(f"$ [OK] {a['name']} -> {a['npy']}")