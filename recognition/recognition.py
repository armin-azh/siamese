from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import configparser
import cv2
import numpy as np
from .utils import load_model, parse_status, FPS, Timer
from .preprocessing import normalize_input, cvt_to_gray
from .cluster import k_mean_clustering
from .distance import bulk_cosine_similarity, bulk_cosine_similarity_v2
from settings import BASE_DIR
from face_detection.detector import FaceDetector
import tensorflow as tf
from tensorflow.keras.models import load_model as h5_load
from database.component import ImageDatabase, parse_test_dir
from settings import BASE_DIR
from PIL import Image
from sklearn import preprocessing
from datetime import datetime
from tqdm import tqdm


def face_recognition(args):
    """
    argument from arg parser
    :param args:
    :return:
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    print("$ On {}".format(physical_devices[0].name))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print(f"$ {parse_status(args)} recognition mode ...")

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    detector_conf = conf['Detector']
    gallery_conf = conf['Gallery']
    default_conf = conf['Default']

    database = ImageDatabase(db_path=gallery_conf.get("database_path"))

    # check cluster name existence
    if args.cluster:
        if args.cluster_name and database.check_name_exists(args.cluster_name):
            print(f" {args.cluster_name} is exists.")

    if (args.realtime or args.video) and args.eval_method == "cosine":
        print("$ loading embeddings ...")
        embeds, labels = database.bulk_embeddings()
        encoded_labels = preprocessing.LabelEncoder()
        encoded_labels.fit(list(set(labels)))
        labels = encoded_labels.transform(labels)

    with tf.device('/device:gpu:0'):
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

                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                if args.save:
                    filename = os.path.join(BASE_DIR, default_conf.get("save_video"),
                                            datetime.strftime(datetime.now(), '%Y%m%d'))

                    if not os.path.exists(filename):
                        os.makedirs(filename)
                    filename = os.path.join(filename,
                                            parse_status() + datetime.strftime(datetime.now(), '%H%M%S') + ".avi")
                    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), )

                fps = FPS()
                fps.start()
                prev = 0

                while cap.isOpened():
                    delta_time = time.time() - prev
                    ret, frame = cap.read()
                    if not ret:
                        break

                    fps.update()
                    if delta_time > 1. / float(detector_conf['fps']):

                        prev = time.time() - prev
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not args.cluster:
                            faces = list()

                        boxes = []
                        gray_frame = cvt_to_gray(frame) if not args.cluster else frame
                        for face, bbox in detector.extract_faces(gray_frame, frame_width * frame_height):
                            faces.append(normalize_input(face))
                            boxes.append(bbox)
                            if args.cluster:
                                faces_.append(face)
                                print("#", end="")

                        if not args.cluster:
                            faces = np.array(faces)

                        if (args.video or args.realtime) and (faces.shape[0] > 0):
                            feed_dic = {phase_train: False, input_plc: faces}
                            embedded_array = sess.run(embeddings, feed_dic)

                            if args.eval_method == 'cosine':
                                dists = bulk_cosine_similarity(embedded_array, embeds)
                                bs_similarity_idx = np.argmin(dists, axis=1)
                                bs_similarity = dists[np.arange(len(bs_similarity_idx)), bs_similarity_idx]
                                pred_labels = np.array(labels)[bs_similarity_idx]
                                for i in range(len(pred_labels)):
                                    x, y, w, h = boxes[i]
                                    status = encoded_labels.inverse_transform([pred_labels[i]]) if bs_similarity[
                                                                                                       i] < float(
                                        default_conf.get("similarity_threshold")) else 'unrecognised'
                                    color = (243, 32, 19) if status == 'unrecognised' else (0, 255, 0)
                                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

                                    cv2.putText(frame,
                                                "{} {:.2f}".format(status[0],
                                                                   bs_similarity[i]),
                                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            else:
                                break

                            if not args.cluster:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                cv2.imshow(parse_status(args), frame)
                                cv2.imshow('gray', gray_frame)
                        else:
                            if not args.cluster:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                cv2.imshow(parse_status(args), frame)
                                cv2.imshow('gray', gray_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                fps.stop()
                if args.cluster:
                    print('\n$ Cluster embeddings')
                    faces = np.array(faces)
                    if faces.shape[0] > 0:
                        feed_dic = {phase_train: False, input_plc: faces}
                        embedded_array = sess.run(embeddings, feed_dic)
                        clusters = k_mean_clustering(embeddings=embedded_array,
                                                     n_cluster=int(gallery_conf['n_clusters']))
                        database.save_clusters(clusters, faces_, args.cluster_name)

                print("$ fps: {:.2f}".format(fps.fps()))
                print("$ expected fps: {}".format(int(cap.get(cv2.CAP_PROP_FPS))))
                print("$ frame width {}".format(frame_width))
                print("$ frame height {}".format(frame_height))
                print("$ elapsed time: {:.2f}".format(fps.elapsed()))

            cap.release()
            cv2.destroyAllWindows()


def face_recognition_on_keras(args):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print(f"$ {parse_status(args)} recognition mode ...")

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    detector_conf = conf['Detector']
    gallery_conf = conf['Gallery']
    default_conf = conf['Default']

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
        labels = encoded_labels.transform(labels)

    model = h5_load(model_conf.get("facenet_keras"))

    detector = FaceDetector(sess=None)
    print("$ MTCNN face detector has been loaded.")

    cap = cv2.VideoCapture(0 if args.realtime else args.video_file)

    if args.cluster:
        faces = list()
        faces_ = list()
        print("$ ", end='')

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if args.save:
        filename = os.path.join(BASE_DIR, default_conf.get("save_video"),
                                datetime.strftime(datetime.now(), '%Y%m%d'))

        if not os.path.exists(filename):
            os.makedirs(filename)
        filename = os.path.join(filename, parse_status() + datetime.strftime(datetime.now(), '%H%M%S') + ".avi")
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), )

    fps = FPS()
    fps.start()
    prev = 0

    while cap.isOpened():
        delta_time = time.time() - prev
        ret, frame = cap.read()

        if not ret:
            break

        if delta_time > 1. / float(detector_conf['fps']):
            fps.update()
            prev = time.time() - prev
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not args.cluster:
                faces = list()

            boxes = []
            for face, bbox in detector.extract_faces(frame, frame_width * frame_height):
                faces.append(normalize_input(face))
                boxes.append(bbox)
                if args.cluster:
                    faces_.append(face)
                    print("#", end="")

            if not args.cluster:
                faces = np.array(faces)

            if (args.video or args.realtime) and (faces.shape[0] > 0):
                embedded_array = model.predict(faces)

                if args.eval_method == 'cosine':
                    dists = bulk_cosine_similarity_v2(embedded_array, embeds)
                    bs_similarity_idx = np.argmin(dists, axis=1)
                    bs_similarity = dists[np.arange(len(bs_similarity_idx)), bs_similarity_idx]
                    pred_labels = np.array(labels)[bs_similarity_idx]
                    for i in range(len(pred_labels)):
                        x, y, w, h = boxes[i]
                        status = encoded_labels.inverse_transform([pred_labels[i]]) if bs_similarity[
                                                                                           i] < float(
                            default_conf.get("similarity_threshold")) else 'unrecognised'
                        color = (243, 32, 19) if status == 'unrecognised' else (0, 255, 0)
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

                        cv2.putText(frame,
                                    "{} {:.2f}".format(status[0],
                                                       bs_similarity[i]),
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(parse_status(args), frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(parse_status(args), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    if args.cluster:
        print('\n$ Cluster embeddings')
        faces = np.array(faces)
        print(faces)
        if faces.shape[0] > 0:
            embedded_array = model.predict(faces)
            clusters = k_mean_clustering(embeddings=embedded_array,
                                         n_cluster=int(gallery_conf['n_clusters']))
            database.save_clusters(clusters, faces_, args.cluster_name)

    print("$ fps: {:.2f}".format(fps.fps()))
    print("$ elapsed time: {:.2f}".format(fps.elapsed()))


def test_recognition(args):
    """
    test prob set on gallery set
    :param args:
    :return: None
    """
    # memory growth
    tf.compat.v1.disable_eager_execution()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if physical_devices:
        print(f"$ {len(physical_devices)} Devices has been detected.")

    conf = configparser.ConfigParser()
    conf.read(os.path.join(BASE_DIR, "conf.ini"))
    model_conf = conf['Model']
    gallery_conf = conf['Gallery']
    test_dir = args.test_dir

    print("$ Test mode ...")
    print(f"$ Test directory {test_dir}")

    test_images, *data = parse_test_dir(test_dir)
    test_labels, ids = data
    test_label_encoder = preprocessing.LabelEncoder()
    test_label_encoder.fit(ids)

    print(f"$ total classes: {len(ids)}")
    print(f"$ total images: {len(test_labels)}")

    print("$ loading embeddings ...")

    database = ImageDatabase(db_path=gallery_conf.get("database_path"))
    gallery_embeds, gallery_labels = database.bulk_embeddings()
    encoded_labels = preprocessing.LabelEncoder()
    encoded_labels.fit(list(set(gallery_labels)))
    gallery_labels = encoded_labels.transform(gallery_labels)

    print("$ loading images in memory")
    test_images_memory = list()
    for im in tqdm(test_images):
        test_images_memory.append(im.read_image_file(grayscale=False, resize=True))

    test_images_memory = np.array(test_images_memory)

    timer = Timer()
    with tf.device('/device:gpu:0'):
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                print(f"$ Initializing computation graph with {model_conf.get('facenet')} pretrained model.")
                load_model(os.path.join(BASE_DIR, model_conf.get('facenet')))
                print("$ Model has been loaded.")

                input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                feed_dic = {phase_train: False, input_plc: test_images_memory}
                timer.start()
                test_embeddings = sess.run(embeddings, feed_dic)
                print(f"$ embeddings created at {timer.end()}")

                test_embeddings = np.array(test_embeddings)
                timer.start()
                dists = bulk_cosine_similarity(test_embeddings, gallery_embeds)
                print(f"$ distances created at {timer.end()}")

                dists = np.array(dists)
                bs_similarity_idx = np.argmin(dists, axis=1)

                accuracy = np.mean(np.equal(test_label_encoder.transform(test_labels), np.array(gallery_labels)[bs_similarity_idx]))
                print(f"$ accuracy {accuracy*100}")