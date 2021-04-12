from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import configparser
import cv2
import numpy as np
from .utils import load_model
from .preprocessing import normalize_input
from settings import BASE_DIR
from face_detection.detector import FaceDetector
import tensorflow as tf


def face_recognition(args):
    """
    argument from arg parser
    :param args:
    :return:
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print("$ Realtime recognition mode ...") if args.realtime else print("$ Video recognition mode ...")

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    detector_conf = conf['Detector']

    prev = time.time()

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            print(f"$ Initializing computation graph with {model_conf.get('facenet')} pretrained model.")
            load_model(os.path.join(BASE_DIR, model_conf.get('facenet')))
            print("$ Model has been loaded.")
            input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            detector = FaceDetector(sess=sess)
            print("$ MTCNN face detector has been loaded.")

            cap = cv2.VideoCapture(0 if args.realtime else args.video_file)

            while cap.isOpened():
                delta_time = time.time() - prev
                ret, frame = cap.read()

                if not ret:
                    break

                if delta_time > 1. / float(detector_conf['fps']):
                    prev = time.time() - prev

                    faces = list()
                    for face in detector.extract_faces(frame):
                        faces.append(normalize_input(face))

                    faces = np.array(faces)

                    print(faces.shape)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
