from unittest import TestCase
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2

# from models
from .base import BaseModel
from ._face_detector import FaceDetector
from ._recognizer import FaceNetModel
from ._hpe import HeadPoseEstimatorModel
from v2.core.nomalizer import GrayScaleConvertor

# exceptions
from v2.core.exceptions import *

from settings import BASE_DIR
from settings import MODEL_CONF
from settings import HPE_CONF


class BaseModelTestCase(TestCase):
    def setUp(self) -> None:
        self.tf_model = Path("G:\\Documents\\Project\\facerecognition\\v2\\core\\network\\data\\1225.pb")
        self.keras_model = Path("G:\\Documents\\Project\\facerecognition\\v2\\core\\network\\data\\1224.h5")
        self.un_model = Path("./data/1226.hp")

    def test_create_tf_model(self):
        model = BaseModel(self.tf_model)
        self.assertEqual(model.model_type, "tf")

    def test_create_keras_model(self):
        model = BaseModel(self.keras_model)
        self.assertEqual(model.model_type, "keras")

    def test_create_invalid_file_name(self):
        with self.assertRaises(UnknownModelFileError):
            _ = BaseModel(self.un_model)

    def test_create_keras_model_without_passing_session(self):
        model = BaseModel(self.keras_model)
        with self.assertRaises(SessionIsNotSetError):
            model.load_model()

    def test_load_keras_model(self):
        model = BaseModel(self.keras_model)
        sess = tf.compat.v1.Session()
        model.load_model(session=sess)
        self.assertTrue(len(model.inputs) > 0)
        t_shape = tf.TensorShape([None, 64, 64, 1])
        model.inputs[0][1].assert_is_compatible_with(t_shape)

        self.assertTrue(len(model.outputs) > 0)

    def test_load_tf_model(self):
        model = BaseModel(self.tf_model)
        model.set_inputs_name(["x:0"])
        model.set_outputs_name(["Identity:0"])
        model.load_model()

        self.assertTrue(len(model.inputs) > 0)
        t_shape = tf.TensorShape([None, 1])
        model.outputs[0][1].assert_is_compatible_with(t_shape)


class FaceDetectorTestCase(TestCase):
    def test_create_face_detector_class_and_disabled_methods(self):
        threshold = [0.8, 0.8, 0.9]
        scale_factor = 0.8
        min_size = 20
        face_detector = FaceDetector(min_face=min_size, scale_factor=scale_factor, stages_threshold=threshold,
                                     name="mtcnn")
        self.assertTrue(isinstance(face_detector, FaceDetector))

        with self.assertRaises(DisableMethodWarning):
            _ = face_detector.inputs

        with self.assertRaises(DisableMethodWarning):
            _ = face_detector.outputs

        with self.assertRaises(DisableMethodWarning):
            face_detector.set_inputs_name([""])

        with self.assertRaises(DisableMethodWarning):
            face_detector.set_outputs_name([""])

    def test_create_face_detector_class_with_session(self):
        # memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        threshold = [0.8, 0.8, 0.9]
        scale_factor = 0.8
        min_size = 20
        face_detector = FaceDetector(min_face=min_size, scale_factor=scale_factor, stages_threshold=threshold,
                                     name="mtcnn")
        with tf.compat.v1.Session() as sess:
            face_detector.load_model(session=sess)

    def test_create_face_detector_class_without_session(self):
        # memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        threshold = [0.8, 0.8, 0.9]
        scale_factor = 0.8
        min_size = 20
        face_detector = FaceDetector(min_face=min_size, scale_factor=scale_factor, stages_threshold=threshold,
                                     name="mtcnn")
        with self.assertRaises(SessionIsNotSetError):
            face_detector.load_model()

    def test_extract_face_detector(self):
        # memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        threshold = [0.8, 0.8, 0.9]
        scale_factor = 0.8
        min_size = 20
        face_detector = FaceDetector(min_face=min_size, scale_factor=scale_factor, stages_threshold=threshold,
                                     name="mtcnn")
        with tf.compat.v1.Session() as sess:
            face_detector.load_model(session=sess)

            im = cv2.imread("G:\\Documents\\Project\\facerecognition\\v2\\core\\network\\data\\celeb.jpg")
            _, _ = face_detector.extract(im)


class FaceNetModelTestCase(TestCase):
    def setUp(self) -> None:
        self._model_path = Path(BASE_DIR).joinpath(MODEL_CONF.get("facenet"))
        self._image = cv2.resize(GrayScaleConvertor().normalize(
            cv2.imread(str(Path(BASE_DIR).joinpath("v2/core/network/data/celeb.jpg"))), channel="full"), (160, 160))

    def test_create_face_net_model(self):
        model = FaceNetModel(model_path=self._model_path, name="FaceNet")
        model.load_model()
        self.assertEqual(model.model_type, "tf")

    def test_create_face_net_run(self):
        model = FaceNetModel(model_path=self._model_path, name="FaceNet")
        in_ims = np.expand_dims(self._image, axis=0)
        with tf.compat.v1.Session() as sess:
            model.load_model()
            emb = model.get_embeddings(sess, input_im=in_ims)
            self.assertEqual(emb.shape, (1, 512))

    def test_create_face_net_run_with_incompatible_shape(self):
        model = FaceNetModel(model_path=self._model_path, name="FaceNet")
        in_ims = np.random.random((160, 160, 3))
        with tf.compat.v1.Session() as sess:
            model.load_model()
            with self.assertRaises(InCompatibleDimError):
                _ = model.get_embeddings(sess, input_im=in_ims)


class HeadPoseEstimatorTestCase(TestCase):
    def setUp(self) -> None:
        self._im_path = Path(BASE_DIR).joinpath("v2/core/network/data/celeb.jpg")
        self._im = cv2.cvtColor(cv2.imread(str(self._im_path)), cv2.COLOR_BGR2GRAY)

        self._bounding_boxes = np.array([[54, 78, 75, 98], [12, 45, 32, 65]])

        self._model_path = Path(BASE_DIR).joinpath(HPE_CONF.get("model"))
        self._img_norm = (float(HPE_CONF.get("im_norm_mean")), float(HPE_CONF.get("im_norm_var")))
        self._tilt_norm = (float(HPE_CONF.get("tilt_norm_mean")), float(HPE_CONF.get("tilt_norm_var")))
        self._pat_norm = (float(HPE_CONF.get("pan_norm_mean")), float(HPE_CONF.get("pan_norm_var")))
        self._rescale = float(HPE_CONF.get("rescale"))
        self._hpe_conf = (
            float(HPE_CONF.get("pan_left")),
            float(HPE_CONF.get("pan_right")),
            float(HPE_CONF.get("tilt_up")),
            float(HPE_CONF.get("tilt_down"))
        )

    def test_create_hpe_a_object(self):
        model = HeadPoseEstimatorModel(model_path=self._model_path,
                                       img_norm=self._img_norm,
                                       tilt_norm=self._tilt_norm,
                                       pan_norm=self._pat_norm,
                                       rescale=self._rescale,
                                       conf=self._hpe_conf)
        self.assertEqual(model._name, model.__class__.__name__)
        self.assertEqual(model.model_type, "tf")
        with tf.compat.v1.Session() as sess:
            model.load_model()
            self.assertEqual(model.inputs[0][0], "x:0")
            self.assertEqual(model.outputs[0][0], "Identity:0")

    def test_predict_poses(self):
        model = HeadPoseEstimatorModel(model_path=self._model_path,
                                       img_norm=self._img_norm,
                                       tilt_norm=self._tilt_norm,
                                       pan_norm=self._pat_norm,
                                       rescale=self._rescale,
                                       conf=self._hpe_conf)

        with tf.compat.v1.Session() as sess:
            model.load_model()

            poses = model.estimate_poses(session=sess,
                                         input_im=self._im,
                                         boxes=self._bounding_boxes,
                                         )
            self.assertEqual(poses.shape, (2, 2))

    def test_predict_poses_and_validate_angles(self):
        model = HeadPoseEstimatorModel(model_path=self._model_path,
                                       img_norm=self._img_norm,
                                       tilt_norm=self._tilt_norm,
                                       pan_norm=self._pat_norm,
                                       rescale=self._rescale,
                                       conf=self._hpe_conf)

        with tf.compat.v1.Session() as sess:
            model.load_model()

            poses = model.estimate_poses(session=sess,
                                         input_im=self._im,
                                         boxes=self._bounding_boxes,
                                         )
            idx, idx_c = model.validate_angle(poses)
            self.assertTrue((len(idx) > 0 or len(idx_c) > 0))

    def test_predict_poses_and_validate_angles_with_empty_input(self):
        model = HeadPoseEstimatorModel(model_path=self._model_path,
                                       img_norm=self._img_norm,
                                       tilt_norm=self._tilt_norm,
                                       pan_norm=self._pat_norm,
                                       rescale=self._rescale,
                                       conf=self._hpe_conf)

        with tf.compat.v1.Session() as sess:
            model.load_model()

            poses = np.empty((0, 2))
            idx, idx_c = model.validate_angle(poses)
            self.assertEqual(idx.shape, (0, 2))
            self.assertEqual(idx_c.shape, (0, 2))
