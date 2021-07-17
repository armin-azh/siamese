from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import pathlib
import time
import configparser
import cv2
import numpy as np
from recognition.utils import load_model, parse_status, FPS, Timer
from recognition.preprocessing import normalize_input, cvt_to_gray
from recognition.cluster import k_mean_clustering
from recognition.distance import bulk_cosine_similarity, bulk_cosine_similarity_v2
from settings import BASE_DIR
from face_detection.detector import FaceDetector
from tracker.policy import Policy, PolicyTracker
from database.sync import parse_person_id_dictionary
import tensorflow as tf
from tensorflow.keras.models import load_model as h5_load
from database.component import ImageDatabase, parse_test_dir
from database.component import inference_db
from motion_detection.component import BSMotionDetection
from face_detection.tracker import Tracker, KalmanFaceTracker
from face_detection.utils import draw_face
from tracker import TrackerList, MATCHED, UN_MATCHED
from settings import BASE_DIR, GALLERY_CONF, DEFAULT_CONF, MODEL_CONF, GALLERY_ROOT, CAMERA_MODEL_CONF, DETECTOR_CONF, \
    SERVER_CONF
from sklearn import preprocessing
from datetime import datetime
from tools.logger import Logger
from tools.logger.logger import ExeLogger
from stream.source import OpencvSource
import socket
from settings import (COLOR_WARN,
                      COLOR_DANG,
                      COLOR_SUCCESS,
                      TRACKER_CONF,
                      SUMMARY_LOG_DIR,
                      SOURCE_CONF)
from uuid import uuid1

# serializer
from .serializer import face_serializer


def recognition_serve(args):
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

                _source = 0
                _source_name = ""
                if args.realtime and args.proto == "rtsp":
                    _source = SOURCE_CONF.get("cam_1")
                    _source_name = "rtsp_video_cam_1"
                elif args.video_file:
                    _source = args.video_file
                    _source_name = "local_video_cam"
                cap = OpencvSource(src=_source, name=_source_name, width=int(CAMERA_MODEL_CONF.get("width")),
                                   height=int(CAMERA_MODEL_CONF.get("height")))

                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                if not args.cluster:
                    tk = TrackerList(float(TRACKER_CONF.get("max_modify_time")),
                                     int(TRACKER_CONF.get("max_frame_conf")))
                prev = 1
                proc_timer = Timer()
                while cap.isOpened():
                    proc_timer.start()
                    ret, frame = cap.read()
                    cur = time.time()
                    delta_time = cur - prev
                    if not ret:
                        break

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
                                            color = COLOR_SUCCESS
                                            status = [""]


                            else:
                                break

                            tk.modify()


def recognition_serv_2(args):
    # database
    database = ImageDatabase(db_path=GALLERY_CONF.get("database_path"))
    embeds, labels = database.bulk_embeddings()
    encoded_labels = preprocessing.LabelEncoder()
    encoded_labels.fit(list(set(labels)))
    labels = encoded_labels.transform(labels)
    person_ids = parse_person_id_dictionary()

    # memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tracker = PolicyTracker(max_life_time=float(TRACKER_CONF.get("max_modify_time")),
                            max_conf=int(TRACKER_CONF.get("max_frame_conf")))

    global_unrecognized_cnt = 0

    address = (SERVER_CONF.get("UDP_HOST"), int(SERVER_CONF.get("UDP_PORT")))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    face_save_path = pathlib.Path(SERVER_CONF.get("face_save_path")).joinpath(SERVER_CONF.get("face_folder"))
    if not face_save_path.exists():
        face_save_path.mkdir(parents=True)

    # computation graph
    with tf.device('/device:gpu:0'):
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                load_model(os.path.join(BASE_DIR, MODEL_CONF.get('facenet')))
                input_plc = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

                # face detector
                detector = FaceDetector(sess=sess)

                _source = 0
                _source_name = ""
                if args.realtime and args.proto == "rtsp":
                    _source = SOURCE_CONF.get("cam_1")
                    _source_name = "rtsp_video_cam_1"

                cap = OpencvSource(src=_source, name=_source_name, width=int(CAMERA_MODEL_CONF.get("width")),
                                   height=int(CAMERA_MODEL_CONF.get("height")))

                prev = 1
                while cap.isOpened():
                    cur = time.time()
                    delta_time = cur - prev

                    ret, frame = cap.read()

                    if not ret:
                        break

                    if delta_time > 1. / float(DETECTOR_CONF['fps']):
                        prev = cur

                        serial_event = []

                        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gray_frame = cvt_to_gray(frame_)

                        boxes = []
                        faces = []
                        for face, bbox in detector.extract_faces(gray_frame, int(CAMERA_MODEL_CONF.get("width")),
                                                                 int(CAMERA_MODEL_CONF.get("height"))):
                            faces.append(normalize_input(face))
                            boxes.append(bbox)

                        faces = np.array(faces)

                        cvt_frame = cv2.cvtColor(frame_.copy(), cv2.COLOR_RGB2BGR)

                        if faces.shape[0] > 0:
                            feed_dic = {phase_train: False, input_plc: faces}
                            embedded_array = sess.run(embeddings, feed_dic)
                            dists = bulk_cosine_similarity(embedded_array, embeds)
                            bs_similarity_idx = np.argmin(dists, axis=1)
                            bs_similarity = dists[np.arange(len(bs_similarity_idx)), bs_similarity_idx]
                            pred_labels = np.array(labels)[bs_similarity_idx]
                            for i in range(len(pred_labels)):
                                uu_ = uuid1()
                                file_name_ = uu_.hex + ".jpg"
                                save_path = face_save_path.joinpath(file_name_)
                                x, y, w, h = boxes[i]
                                status = encoded_labels.inverse_transform([pred_labels[i]]) if bs_similarity[
                                                                                                   i] < float(
                                    DEFAULT_CONF.get("similarity_threshold")) else ['unrecognised']

                                print(status[0])

                                try:

                                    if status[0] == "unrecognised":
                                        if global_unrecognized_cnt == int(TRACKER_CONF.get("unrecognized_counter")):
                                            now_ = datetime.now()
                                            serial_ = face_serializer(timestamp=int(now_.timestamp()) * 1000,
                                                                      person_id=None,
                                                                      camera_id=None,
                                                                      image_path=file_name_)

                                            serial_event.append(serial_)
                                            cv2.imwrite(str(save_path), cvt_frame[y:y + h, x:x + w])
                                            global_unrecognized_cnt = 0
                                        else:
                                            global_unrecognized_cnt += 1

                                    else:
                                        tk_ = tracker(name=status[0])

                                        if tk_.status == Policy.STATUS_CONF and not tk_.mark and status[0]:
                                            tk_.mark = True
                                            now_ = datetime.now()
                                            id_ = person_ids.get(status[0])

                                            if id_ is not None:
                                                serial_ = face_serializer(timestamp=int(now_.timestamp() * 1000),
                                                                          person_id=id_,
                                                                          camera_id=None,
                                                                          image_path=file_name_)

                                                # serial_ = face_serializer(timestamp=now_.timestamp(),
                                                #                           person_id=None,
                                                #                           camera_id=None,
                                                #                           image_path=str(save_path))

                                                serial_event.append(serial_)

                                                cv2.imwrite(str(save_path), cvt_frame[y:y + h, x:x + w])

                                except:
                                    print("Record Drop")

                        if serial_event:
                            json_obj = json.dumps({"data": serial_event})
                            sock.sendto(json_obj.encode(), address)
                            print(json_obj + " Send to " + f"{address[0]}:{address[1]}")
