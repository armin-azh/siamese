from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import pathlib
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
from database.component import inference_db
from motion_detection.component import BSMotionDetection
from face_detection.tracker import Tracker, KalmanFaceTracker
from face_detection.utils import draw_face
from tracker import TrackerList, MATCHED, UN_MATCHED
from settings import BASE_DIR, GALLERY_CONF, DEFAULT_CONF, MODEL_CONF, GALLERY_ROOT
from PIL import Image
from sklearn import preprocessing
from datetime import datetime
from tqdm import tqdm
from tools.system import system_status
from tools.logger import Logger
from settings import (COLOR_WARN,
                      COLOR_DANG,
                      COLOR_SUCCESS,
                      TRACKER_CONF,
                      SUMMARY_LOG_DIR)


def face_recognition(args):
    """
    argument from arg parser
    :param args:
    :return:
    """
    log_subdir = SUMMARY_LOG_DIR.joinpath(datetime.strftime(datetime.now(), '%Y-%m-%d(%H-%M-%S)'))
    logger = Logger(log_dir=log_subdir)
    if not log_subdir.exists():
        log_subdir.mkdir(parents=True)

    physical_devices = tf.config.list_physical_devices('GPU')
    logger.info("$ On {}".format(physical_devices[0].name))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logger.info(f"$ {parse_status(args)} recognition mode ...")

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
            logger.info(f" {args.cluster_name} is exists.")

    if (args.realtime or args.video) and args.eval_method == "cosine":
        if not database.is_db_stable():
            logger.info("$ database is not stable, build npy file or no one is registered")
            sys.exit()
        else:
            logger.info("$ database is stable")
        logger.info("$ loading embeddings ...")
        embeds, labels = database.bulk_embeddings()
        encoded_labels = preprocessing.LabelEncoder()
        encoded_labels.fit(list(set(labels)))
        labels = encoded_labels.transform(labels)
    motion_detection = BSMotionDetection()

    with tf.device('/device:gpu:0'):
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                logger.info(f"$ Initializing computation graph with {model_conf.get('facenet')} pretrained model.")
                load_model(os.path.join(BASE_DIR, model_conf.get('facenet')))
                logger.info("$ Model has been loaded.")
                input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                detector = FaceDetector(sess=sess)
                detector_type = detector_conf.get("type")
                logger.info(f"$ {detector_type} face detector has been loaded.")

                cap = cv2.VideoCapture(0 if args.realtime else args.video_file)

                if args.cluster:
                    faces = list()
                    faces_ = list()
                    print("$ ", end='')

                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

                if not args.cluster:
                    fps = FPS()
                    fps.start()
                    tk = TrackerList(float(TRACKER_CONF.get("max_modify_time")),
                                     int(TRACKER_CONF.get("max_frame_conf")))
                prev = 0
                total_proc_time = list()
                proc_timer = Timer()
                while cap.isOpened():
                    proc_timer.start()
                    ret, frame = cap.read()
                    cur = time.time()
                    delta_time = cur - prev
                    if not ret:
                        break

                    if not args.cluster:
                        fps.update()
                    if (delta_time > 1. / float(detector_conf['fps'])) and (
                            motion_detection.has_motion(frame) is not None or args.cluster):

                        prev = cur
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if not args.cluster:
                            faces = list()

                        boxes = []
                        gray_frame = cvt_to_gray(frame) if not args.cluster else frame
                        for face, bbox in detector.extract_faces(gray_frame, frame_width, frame_height):
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
                                        default_conf.get("similarity_threshold")) else ['unrecognised']
                                    if status[0] == "unrecognised":
                                        color = COLOR_DANG
                                    else:
                                        res = tk(name=status)
                                        if res == MATCHED:
                                            color = COLOR_SUCCESS
                                        else:
                                            color = COLOR_WARN
                                            status = [""]

                                    if detector_type == detector.DT_MTCNN:
                                        frame = draw_face(frame, (x, y), (x + w, y + h), 5, 10, color, 1)
                                        # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                                    elif detector_type == detector.DT_RES10:
                                        frame = draw_face(frame, (x, y), (w, h), 5, 10, color, 1)
                                        # frame = cv2.rectangle(frame, (x, y), (w, h), color, 1)

                                    cv2.putText(frame,
                                                "{} {:.2f}".format(status[0],
                                                                   bs_similarity[i]),
                                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            else:
                                break

                            if not args.cluster:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                cv2.imshow(parse_status(args), frame)
                                # cv2.imshow('gray', gray_frame)
                            tk.modify()

                        else:
                            if not args.cluster:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                cv2.imshow(parse_status(args), frame)
                                # cv2.imshow('gray', gray_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    total_proc_time.append(proc_timer.end())

                if not args.cluster:
                    fps.stop()
                if args.cluster:
                    logger.info('\n$ Cluster embeddings')
                    faces = np.array(faces)
                    if faces.shape[0] > 0:
                        feed_dic = {phase_train: False, input_plc: faces}
                        embedded_array = sess.run(embeddings, feed_dic)
                        clusters = k_mean_clustering(embeddings=embedded_array,
                                                     n_cluster=int(gallery_conf['n_clusters']))
                        database.save_clusters(clusters, faces_, args.cluster_name)

                if not args.cluster:
                    fps_rate = fps.fps()
                    fps_elapsed = fps.elapsed()
                    average_iterations = np.array(total_proc_time).mean()
                    logger.info("$ fps: {:.2f}".format(fps_rate))
                    logger.info("$ expected fps: {}".format(int(cap.get(cv2.CAP_PROP_FPS))))
                    logger.info("$ frame width {}".format(frame_width))
                    logger.info("$ frame height {}".format(frame_height))
                    logger.info("$ elapsed time: {:.2f}".format(fps_elapsed))
                    logger.info("$ Average time per iteration: {:.3f}".format(average_iterations))

                    system_result = system_status(args, printed=False)
                    data = {"Scenario": args.scene, "Fps_Rate": fps_rate, "Fps_Elapsed": fps_elapsed,
                            "Average_Time_Per_Iteration": average_iterations, **system_result}
                    logger.log(data)

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

                accuracy = np.mean(
                    np.equal(test_label_encoder.transform(test_labels), np.array(gallery_labels)[bs_similarity_idx]))
                print(f"$ accuracy {accuracy * 100}")


def cluster_faces(args) -> None:
    """
    cluster videos
    :param args:
    :return:
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    print("$ On {}".format(physical_devices[0].name))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    database = ImageDatabase(db_path=GALLERY_ROOT)
    ids = list(database.get_identity_image_paths().keys())

    ls_video = pathlib.Path(BASE_DIR).joinpath(DEFAULT_CONF.get("save_video"))

    print(ls_video)

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:

            print(f"$ Initializing computation graph with {MODEL_CONF.get('facenet')} pretrained model.")
            load_model(os.path.join(BASE_DIR, MODEL_CONF.get('facenet')))
            print("$ Model has been loaded.")

            detector = FaceDetector(sess=sess)

            input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            for v_path in ls_video.glob("*.avi"):

                filename = v_path.stem

                if filename.find("done") == -1:

                    if filename in ids:
                        print(f"[WARN] This id is exists {filename}")
                        continue
                    else:
                        cap = cv2.VideoCapture(str(v_path))

                        f_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        f_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                        n_faces = list()
                        faces = list()

                        cnt = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            for face, _ in detector.extract_faces(frame, f_w, f_h):
                                if cnt % 10 == 0 and cnt > 0:
                                    print("#", end="")
                                n_faces.append(normalize_input(face))
                                faces.append(face)
                        cap.release()

                        n_faces = np.array(n_faces)
                        if n_faces.shape[0] > 0:
                            feed_dic = {phase_train: False, input_plc: n_faces}
                            embedded_array = sess.run(embeddings, feed_dic)
                            clusters = k_mean_clustering(embeddings=embedded_array,
                                                         n_cluster=int(GALLERY_CONF.get("n_clusters")))
                            database.save_clusters(clusters, faces, filename.title())
                            v_path.rename(v_path.parent.joinpath(v_path.stem + "_done" + v_path.suffix))

                else:
                    print(f"[INFO] this file is clustered {filename}")

    inference_db(args)
    print("$ finished")


def face_recognition_kalman(args):
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

    if (args.realtime or args.video) and args.eval_method == "cosine":
        print("$ loading embeddings ...")
        embeds, labels = database.bulk_embeddings()
        encoded_labels = preprocessing.LabelEncoder()
        encoded_labels.fit(list(set(labels)))
        labels = encoded_labels.transform(labels)
    motion_detection = BSMotionDetection()

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
                detector_type = detector_conf.get("type")
                print(f"$ {detector_type} face detector has been loaded.")

                cap = cv2.VideoCapture(0 if args.realtime else args.video_file)

                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                fps = FPS()
                fps.start()
                prev = 0
                total_proc_time = list()
                proc_timer = Timer()

                tracker = Tracker()
                while cap.isOpened():
                    proc_timer.start()
                    ret, frame = cap.read()
                    cur = time.time()
                    delta_time = cur - prev
                    if not ret:
                        break

                    fps.update()
                    if (delta_time > 1. / float(detector_conf['fps'])) and (
                            motion_detection.has_motion(frame) is not None):

                        prev = cur
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        faces = []
                        boxes = []

                        gray_frame = cvt_to_gray(frame) if not args.cluster else frame
                        for face, bbox in detector.extract_faces(gray_frame, frame_width, frame_height):
                            faces.append(normalize_input(face))
                            boxes.append(bbox)

                        boxes = np.array(boxes)
                        faces = np.array(faces)
                        un_matches, tracker_indexes = tracker.find_relative_boxes(boxes)

                        if un_matches is not None:
                            boxes = boxes[un_matches[0], :]
                            faces = faces[un_matches[0], ...]

                        if tracker_indexes is not None:

                            for idx in tracker_indexes:
                                x, y, w, h = tracker.get_tracker_current_state(idx).reshape((4,))
                                x = int(x)
                                y = int(y)
                                w = int(w)
                                h = int(h)
                                frame = draw_face(frame, (x, y), (x + w, y + h), 5, 10, COLOR_SUCCESS, 1)
                                cv2.putText(frame, "{}".format(tk.name),
                                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SUCCESS, 1)

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

                                    if status != "unrecognised":
                                        tk = tracker.add_new_tracker(status, np.array(boxes[i]))
                                        tk.predict()
                                        tk.correction(boxes[i])

                                        # print(tk.status)
                                        color = COLOR_SUCCESS if tk.status == KalmanFaceTracker.STATUS_MATCHED else COLOR_WARN
                                    else:
                                        color = COLOR_DANG

                                    if detector_type == detector.DT_MTCNN:
                                        frame = draw_face(frame, (x, y), (x + w, y + h), 5, 10, color, 1)
                                        # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                                    elif detector_type == detector.DT_RES10:
                                        frame = draw_face(frame, (x, y), (w, h), 5, 10, color, 1)
                                        # frame = cv2.rectangle(frame, (x, y), (w, h), color, 1)

                                    # cv2.putText(frame,
                                    #             "{} {:.2f}".format(status[0],
                                    #                                bs_similarity[i]),
                                    #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    text = status[0] if tk.status != KalmanFaceTracker.STATUS_UNMATCHED else ""
                                    cv2.putText(frame, "{}".format(text),
                                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            else:
                                break

                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imshow(parse_status(args), frame)
                            # cv2.imshow('gray', gray_frame)
                            # print(f"Tracker: {tracker.number_of_trackers}")
                            # tracker.update()
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            cv2.imshow(parse_status(args), frame)
                            # cv2.imshow('gray', gray_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    total_proc_time.append(proc_timer.end())
                fps.stop()

                print("$ fps: {:.2f}".format(fps.fps()))
                print("$ expected fps: {}".format(int(cap.get(cv2.CAP_PROP_FPS))))
                print("$ frame width {}".format(frame_width))
                print("$ frame height {}".format(frame_height))
                print("$ elapsed time: {:.2f}".format(fps.elapsed()))
                print("$ Average time per iteration: {:.3f}".format(np.array(total_proc_time).mean()))

            cap.release()
            cv2.destroyAllWindows()
