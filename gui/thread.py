from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import configparser

import time
import cv2
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5 import QtGui

from face_detection.detector import FaceDetector
from database.component import ImageDatabase
from recognition.preprocessing import normalize_input, cvt_to_gray
from recognition.cluster import k_mean_clustering
from recognition.utils import load_model
from recognition.distance import bulk_cosine_similarity
from settings import BASE_DIR, GALLERY_CONF, MODEL_CONF, DETECTOR_CONF, DEFAULT_CONF
from face_detection.utils import draw_face


class ClusterThread(QThread):
    """
    thread for clustering faces
    """
    TM_VIDEO_PATH = './temp/xs1_tiger.avi'
    CLUSTER_NAME = None

    cluster_signal = pyqtSignal()

    @property
    def cluster_name(self):
        return self.CLUSTER_NAME

    @cluster_name.setter
    def cluster_name(self, name):
        self.CLUSTER_NAME = name

    def run(self) -> None:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        base_path = pathlib.Path(BASE_DIR)

        conf = configparser.ConfigParser()
        conf.read(base_path.joinpath('conf.ini'))
        model_conf = conf['Model']
        gallery_conf = conf['Gallery']
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                load_model(base_path.joinpath(model_conf.get('facenet')))
                input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                detector = FaceDetector(sess=sess)

                cap = cv2.VideoCapture(self.TM_VIDEO_PATH)

                f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                f_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                n_faces = list()
                faces = list()
                while cap.isOpened():

                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    for face, _ in detector.extract_faces(frame, f_w, f_h):
                        n_faces.append(normalize_input(face))
                        faces.append(face)
                cap.release()

                n_faces = np.array(n_faces)
                faces = np.array(faces)
                if faces.shape[0] > 0:
                    feed_dic = {phase_train: False, input_plc: n_faces}
                    embedded_array = sess.run(embeddings, feed_dic)
                    clusters = k_mean_clustering(embeddings=embedded_array,
                                                 n_cluster=int(gallery_conf['n_clusters']))

                    print(clusters)
        self.cluster_signal.emit()
        self.quit()


class VideoSteamerThread(QThread):
    """
    thread for get video stream from webcam
    """
    image_update = pyqtSignal(QtGui.QImage)
    frame_update = pyqtSignal(np.ndarray)
    thread_active = None

    record_on = False

    def run(self):
        self.thread_active = True
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

        f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        x = int(f_w / 2)
        y = int(f_h / 2)

        w = 120
        h = 120

        while self.thread_active:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_2 = frame
                frame = cv2.resize(frame, (640, 480))
                qt_frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)

                frame = draw_face(frame, (x - w, y - h), (x + w, y + h), 10, 20, (0, 204, 0), 2)

                if self.record_on:
                    cv2.putText(frame, 'REC', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                self.image_update.emit(qt_frame)
                self.frame_update.emit(frame_2)
        cap.release()

    def stop(self):
        """
        stop and release webcam
        :return:
        """
        self.thread_active = False
        self.quit()
