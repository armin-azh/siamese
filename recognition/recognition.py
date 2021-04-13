from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import configparser
import cv2
import numpy as np
from .utils import load_model, parse_status
from .preprocessing import normalize_input
from .cluster import k_mean_clustering
from .distance import bulk_cosine_similarity
from settings import BASE_DIR
from face_detection.detector import FaceDetector
import tensorflow as tf
from database.component import ImageDatabase
from settings import BASE_DIR
from PIL import Image
from sklearn import preprocessing


def face_recognition(args):
    """
    argument from arg parser
    :param args:
    :return:
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print(f"$ {parse_status(args)} recognition mode ...")

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    detector_conf = conf['Detector']
    gallery_conf = conf['Gallery']

    prev = time.time()

    database = ImageDatabase(db_path=gallery_conf.get("database_path"))

    # check cluster name existence
    if args.cluster:
        if args.cluster_name and database.check_name_exists(args.cluster_name):
            raise ValueError(f" {args.cluster_name} is exists.")

    if (args.realtime or args.video) and args.eval_method == "cosine":
        print("$ loading embeddings ...")
        embeds, labels = database.bulk_embeddings()
        encoded_labels = preprocessing.LabelEncoder()
        encoded_labels.fit(list(set(labels)))
        print(embeds.shape)

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

            if args.cluster:
                faces = list()
                faces_ = list()
                print("$ ", end='')

            while cap.isOpened():
                delta_time = time.time() - prev
                ret, frame = cap.read()

                if not ret:
                    break

                if delta_time > 1. / float(detector_conf['fps']):
                    prev = time.time() - prev
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not args.cluster:
                        faces = list()

                    for face in detector.extract_faces(frame):
                        faces.append(normalize_input(face))
                        if args.cluster:
                            faces_.append(face)
                            print("#", end="")

                    if not args.cluster:
                        faces = np.array(faces)

                    if (args.video or args.realtime) and (faces.shape[0] > 0):
                        feed_dic = {phase_train: False, input_plc: faces}
                        embedded_array = sess.run(embeddings, feed_dic)

                        if args.eval_method == 'cosine':
                            # dists = bulk_cosine_similarity(embedded_array, db_embeddings)
                            # bs_similarity_idx = np.argmin(dists, axis=1)
                            # bs_similarity = dists[np.arange(len(bs_similarity_idx)), bs_similarity_idx]
                            print("$ [OK]")

                        else:
                            break


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.cluster:
                print('\n$ Cluster embeddings')
                faces = np.array(faces)
                if faces.shape[0] > 0:
                    feed_dic = {phase_train: False, input_plc: faces}
                    embedded_array = sess.run(embeddings, feed_dic)
                    clusters = k_mean_clustering(embeddings=embedded_array,
                                                 n_cluster=int(gallery_conf['n_clusters']))
                    database.save_clusters(clusters, faces_, args.cluster_name)
