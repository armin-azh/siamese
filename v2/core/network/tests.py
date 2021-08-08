from unittest import TestCase
from pathlib import Path
import tensorflow as tf
import cv2

# from models
from .base import BaseModel
from ._face_detector import FaceDetector

# exceptions
from v2.core.exceptions import *


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
