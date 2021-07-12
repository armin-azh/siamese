from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Response
from flask import Flask, jsonify, request
from flask import render_template
from flask_socketio import SocketIO, send, emit
import socketio
import threading
import cv2
from datetime import datetime
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import time
import json
from uuid import uuid1
import requests

# module
from stream.source import OpencvSource
from face_detection.detector import FaceDetector
from face_detection.utils import draw_face
from recognition.distance import bulk_cosine_similarity
from recognition.preprocessing import normalize_input, cvt_to_gray
from recognition.utils import load_model
from database.component import ImageDatabase
from tracker.policy import Policy, PolicyTracker

# settings
from settings import GALLERY_CONF
from settings import BASE_DIR
from settings import MODEL_CONF
from settings import DETECTOR_CONF
from settings import CAMERA_MODEL_CONF
from settings import DEFAULT_CONF
from settings import SOURCE_CONF
from settings import SERVER_CONF
from settings import TRACKER_CONF

# colors
from settings import COLOR_DANG
from settings import COLOR_SUCCESS

# serializer
from .serializer import face_serializer

output_frame = None
output_box_frame = None
lock = threading.Lock()

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'

socket = SocketIO(app, async_mode=None)
socket.init_app(app, cors_allowed_origins="*")

camera_src_name = "default"
src = OpencvSource(src=0, name=camera_src_name, width=640, height=480)

face_save_path = pathlib.Path(SERVER_CONF.get("face_save_path")).joinpath(SERVER_CONF.get("face_folder"))
if not face_save_path.exists():
    face_save_path.mkdir(parents=True)


def get_stream():
    global src, lock, output_frame

    while True:
        _, frame = src.read()

        with lock:
            output_frame = frame.copy()


def generate():
    global output_frame, lock

    while True:

        with lock:
            if output_frame is None:
                continue

            flag, encoded_image = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


def recognition():
    global lock, output_box_frame, face_save_path, output_frame

    # database
    database = ImageDatabase(db_path=GALLERY_CONF.get("database_path"))
    embeds, labels = database.bulk_embeddings()
    encoded_labels = preprocessing.LabelEncoder()
    encoded_labels.fit(list(set(labels)))
    labels = encoded_labels.transform(labels)

    # memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    tracker = PolicyTracker(max_life_time=float(TRACKER_CONF.get("max_modify_time")),
                            max_conf=int(TRACKER_CONF.get("max_frame_conf")))

    global_unrecognized_cnt = 0

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

                prev = 1
                while True:
                    cur = time.time()
                    delta_time = cur - prev

                    if delta_time > 1. / float(DETECTOR_CONF['fps']):
                        prev = cur

                        with lock:

                            if output_frame is None:
                                continue

                            frame_ = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                            gray_frame = cvt_to_gray(frame_)

                            boxes = []
                            faces = []
                            for face, bbox in detector.extract_faces(gray_frame, int(CAMERA_MODEL_CONF.get("width")),
                                                                     int(CAMERA_MODEL_CONF.get("height"))):
                                faces.append(normalize_input(face))
                                boxes.append(bbox)

                            faces = np.array(faces)

                            cvt_frame = cv2.cvtColor(frame_.copy(), cv2.COLOR_RGB2BGR)

                            serial_event = []

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

                                    if status[0] == "unrecognised":
                                        color = COLOR_DANG
                                        if global_unrecognized_cnt == int(TRACKER_CONF.get("unrecognized_counter")):
                                            now_ = datetime.now()
                                            serial_ = face_serializer(timestamp=now_.timestamp(),
                                                                      person_id=None,
                                                                      camera_id=None,
                                                                      image_path=str(save_path))

                                            serial_event.append(serial_)
                                            cv2.imwrite(str(save_path), cvt_frame[y:y + h, x:x + w])
                                            global_unrecognized_cnt = 0
                                        else:
                                            global_unrecognized_cnt += 1

                                    else:
                                        color = COLOR_SUCCESS

                                        tk_ = tracker(name=status[0])

                                        if tk_.status == Policy.STATUS_CONF and not tk_.mark and status[0]:
                                            tk_.mark = True
                                            now_ = datetime.now()
                                            # serial_ = face_serializer(timestamp=now_.timestamp(),
                                            #                           person_id=status[0] if status[
                                            #                                                      0] != "unrecognised" else None,
                                            #                           camera_id=camera_src_name,
                                            #                           image_path=str(save_path))

                                            serial_ = face_serializer(timestamp=now_.timestamp(),
                                                                      person_id=None,
                                                                      camera_id=None,
                                                                      image_path=str(save_path))

                                            serial_event.append(serial_)
                                            cv2.imwrite(str(save_path), cvt_frame[y:y + h, x:x + w])

                                    frame_ = draw_face(frame_, (x, y), (x + w, y + h), 5, 10, color, 1)

                                    frame_ = cv2.putText(frame_,
                                                         "{} {:.2f}".format(status[0],
                                                                            bs_similarity[i]),
                                                         (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        frame_ = cv2.cvtColor(frame_, cv2.COLOR_RGB2BGR)

                        output_box_frame = frame_.copy()

                        if serial_event:
                            json_obj = json.dumps({"data": serial_event})
                            requests.post(SERVER_CONF.get("face_event_url"), json=json_obj)


def stream_recognition():
    """
    realtime detection, this generator yield frame
    :return:
    """
    global output_frame, lock, output_box_frame

    while True:

        with lock:
            if output_box_frame is None:
                continue

            flag, encoded_image = cv2.imencode(".jpg", output_box_frame)

            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


@app.route("/")
def index():
    t_ = datetime.now()
    data = {"project": "Face Recognition",
            "time": t_.timestamp()}
    return jsonify(data)


@app.route("/api/createUUID/")
def create_uuid_handler():
    u_id = uuid1()

    return jsonify(uuid=u_id.hex)


@app.route("/api/stream/demo/")
def demo():
    """
    demo show
    :return:
    """
    return render_template("demo.html")


@app.route("/api/stream/default.mjpg")
def default_streamer():
    """
    default streamer
    :return:
    """
    data_get = generate()
    return Response(data_get,
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stream/recognition.mjpg")
def recognition_streamer():
    data_get = stream_recognition()

    return Response(data_get,
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/test/websocket/")
def test_websocket_handler():
    return render_template("test_websocket.html")


@app.route("/face/app/faceevent/", methods=["POST"])
def test_face_event_handler():
    req = request.json

    print(req)

    return "ok"


@socket.on("face_event_request")
def face_event_handler(data):
    global output_frame, lock, camera_src_name

    # database
    database = ImageDatabase(db_path=GALLERY_CONF.get("database_path"))
    embeds, labels = database.bulk_embeddings()
    encoded_labels = preprocessing.LabelEncoder()
    encoded_labels.fit(list(set(labels)))
    labels = encoded_labels.transform(labels)

    # memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

                prev = 1
                while True:
                    cur = time.time()
                    delta_time = cur - prev

                    if delta_time > 1. / float(DETECTOR_CONF['fps']):
                        prev = cur

                        with lock:

                            if output_frame is None:
                                continue

                            frame_ = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                            gray_frame = cvt_to_gray(frame_)

                            boxes = []
                            faces = []
                            for face, bbox in detector.extract_faces(gray_frame, int(CAMERA_MODEL_CONF.get("width")),
                                                                     int(CAMERA_MODEL_CONF.get("height"))):
                                faces.append(normalize_input(face))
                                boxes.append(bbox)

                            faces = np.array(faces)

                            rec_list = list()

                            if faces.shape[0] > 0:
                                feed_dic = {phase_train: False, input_plc: faces}
                                embedded_array = sess.run(embeddings, feed_dic)
                                dists = bulk_cosine_similarity(embedded_array, embeds)
                                bs_similarity_idx = np.argmin(dists, axis=1)
                                bs_similarity = dists[np.arange(len(bs_similarity_idx)), bs_similarity_idx]
                                pred_labels = np.array(labels)[bs_similarity_idx]
                                for i in range(len(pred_labels)):
                                    x, y, w, h = boxes[i]
                                    status = encoded_labels.inverse_transform([pred_labels[i]]) if bs_similarity[
                                                                                                       i] < float(
                                        DEFAULT_CONF.get("similarity_threshold")) else ['unrecognised']

                                    if status[0] != "unrecognised":
                                        now_ = datetime.now()
                                        serial_ = face_serializer(timestamp=now_.timestamp(),
                                                                  person_id=status[0],
                                                                  camera_id=camera_src_name,
                                                                  image_path="example")
                                        print(f"Person: {status[0]}")
                                        rec_list.append(serial_)

                            print(rec_list)
                            json_obj = json.dumps({"data": rec_list, "signature": "xxx-xxx-xxx"})

                            emit("face_event_response", json_obj)


def run(args):
    global src, camera_src_name

    if args.rtsp:
        camera_src_name = "rtsp_camera_1"
        src = OpencvSource(SOURCE_CONF.get("cam_1"), camera_src_name, 640, 480)

    t = threading.Thread(target=get_stream, args=())
    t.daemon = True
    t.start()

    t2 = threading.Thread(target=recognition, args=())
    t2.daemon = True
    t2.start()

    socket.run(app=app, host=args.host, port=args.port, debug=True, use_reloader=False)
