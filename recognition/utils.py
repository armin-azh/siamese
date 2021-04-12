from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import re


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
