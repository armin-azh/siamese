import os
import re
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf

# exceptions
from v2.core.exceptions import *


class BaseModel:
    __file_suffixes = {
        "tf": [".pb", ".pdtxt"],
        "keras": [".h5", ".model"]
    }

    def __init__(self, model_path: Path, name=None, *args, **kwargs):
        _tm = self.__determine_type(model_path)
        self._model = None
        self._type = None
        if _tm != "unknown":
            self._type = _tm
        else:
            raise UnknownModelFileError(f"{model_path.suffix} is unknown")
        self._name = self.__class__.__name__ if name is None else name
        self._model_path = model_path
        self._inputs = []
        self._outputs = []
        self._inputs_name = []
        self._outputs_name = []

        super(BaseModel, self).__init__(*args, **kwargs)

    def set_inputs_name(self, names: List[str]) -> None:
        self._inputs_name = names

    def set_outputs_name(self, names: List[str]) -> None:
        self._outputs_name = names

    @property
    def model_type(self) -> str:
        return self._type

    @property
    def inputs(self) -> List[Tuple[str, tf.TensorShape]]:
        return [(layer[1], layer[2]) for layer in self._inputs]

    @property
    def outputs(self) -> List[Tuple[str, tf.TensorShape]]:
        return [(layer[1], layer[2]) for layer in self._outputs]

    def __determine_type(self, filename: Path):
        suf = filename.suffix
        if suf in self.__file_suffixes["tf"]:
            return "tf"
        elif suf in self.__file_suffixes["keras"]:
            return "keras"
        else:
            return "unknown"

    def __keras_inference(self, session: tf.compat.v1.Session, model_path: Path):
        tf.compat.v1.keras.backend.set_session(session)
        self._model = tf.keras.models.load_model(str(model_path))
        self._inputs = [(layer, layer.name, layer.shape) for layer in self._model.inputs]
        self._outputs = [(layer, layer.name, layer.shape) for layer in self._model.outputs]

    def __tf_inference(self, model_path: Path, input_map=None):
        with tf.compat.v1.gfile.GFile(str(model_path), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

        self._inputs = [tf.compat.v1.get_default_graph().get_tensor_by_name(layer_name) for layer_name in
                        self._inputs_name]
        self._inputs = [(layer, layer.name, layer.shape) for layer in self._inputs]
        self._outputs = [tf.compat.v1.get_default_graph().get_tensor_by_name(layer_name) for layer_name in
                         self._outputs_name]
        self._outputs = [(layer, layer.name, layer.shape) for layer in self._outputs]

    def load_model(self, **kwargs):

        if self._type == "tf":
            self.__tf_inference(self._model_path)

        else:
            _sess = kwargs.get("session")
            if _sess is None:
                raise SessionIsNotSetError("you should set session on load_model")
            self.__keras_inference(_sess, self._model_path)
