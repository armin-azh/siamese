from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import pathlib
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import re
import numpy as np
from .model import *


def get_model_filenames(model_dir):
    """
    this code has inspired by
    :param model_dir:
    :return:
    """
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None):
    """
    this code has inspired by
    :param model:
    :param input_map:
    :return:
    """
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        meta_file, ckpt_file = get_model_filenames(model_exp)
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.compat.v1.get_default_session(), os.path.join(model_exp, ckpt_file))


def parse_status(args):
    """
    parse status
    :param args:
    :return:
    """
    if args.video:
        return "video"
    elif args.realtime:
        return "realtime"
    else:
        return "clustering"


def get_filename(key, reg_1, reg_2):
    """
    generate new filename
    :param key:
    :param reg_1:
    :param reg_2:
    :return:
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    filename = reg_1.sub('B', filename)

    if reg_2.match(filename):
        filename = filename.replace('Block8', 'Block8_6')

    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def convert_computation_graph_to_keras_model(model_dir, save_dir, lite: bool = True):
    """
    these function convert computation graph to keras model
    :param lite:
    :param model_dir:
    :param save_dir:
    :return:
    """
    npy_weights_dir = os.path.join(save_dir, 'keras/npy_weights')
    weights_dir = os.path.join(save_dir, 'keras/weights')
    o_model_dir = os.path.join(save_dir, 'keras')

    weights_filename = 'pre_trained_face_net_weights.h5'
    model_filename = 'pre_trained_face_net.h5'
    lite_model_filename = 'pre_trained_face_net.tflite'

    if not os.path.exists(npy_weights_dir):
        os.makedirs(npy_weights_dir)

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if not os.path.exists(o_model_dir):
        os.makedirs(o_model_dir)

    re_repeat = re.compile(r'Repeat_[0-9_]*b')
    re_block8 = re.compile(r'Block8_[A-Za-z]')

    ck_filename = os.path.join(model_dir, 'model-20180402-114759.ckpt-275')
    reader = tf.compat.v1.train.NewCheckpointReader(ck_filename)

    for key in reader.get_variable_to_shape_map():
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        path = os.path.join(npy_weights_dir, get_filename(key, re_repeat, re_block8))
        arr = reader.get_tensor(key)
        np.save(path, arr)

    model = InceptionResNetV1()

    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)

    print(f'$ Saving weights {os.path.join(weights_dir, weights_filename)}')
    model.save_weights(os.path.join(weights_dir, weights_filename))
    print(f'$ Saving model {os.path.join(o_model_dir, model_filename)}')
    o_model_path = os.path.join(o_model_dir, model_filename)
    model.save(o_model_path)

    if lite:
        convertor = tf.compat.v1.lite.TocoConverter.from_keras_model_file(o_model_path)
        convertor.post_training_quantize = True
        tf_lite_model = convertor.convert()
        with open(os.path.join(o_model_dir, lite_model_filename), 'wb') as o_file:
            o_file.write(tf_lite_model)

        print(f"$ lite tensorflow model created {os.path.join(o_model_dir, lite_model_filename)}")


class FPS:
    def __init__(self):
        self._start = None
        self._end = None
        self._n_frame = 0

    def start(self):
        self._start = datetime.datetime.now()

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._n_frame += 1

    def elapsed(self):
        return (self._end - self._start).total_seconds()

    def fps(self):
        return self._n_frame / self.elapsed()


class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def end(self):
        end = time.time() - self._start
        self._start = time.time()
        return end
