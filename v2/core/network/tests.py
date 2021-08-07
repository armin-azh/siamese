from unittest import TestCase
from pathlib import Path
import tensorflow as tf

# from models
from .base import BaseModel

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
