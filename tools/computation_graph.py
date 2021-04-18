import tensorflow as tf
from tensorflow.python.platform import gfile


def pb_to_tensorboard(log_dir, model_dir):
    """
    convert computation graph to tensorflow
    :param log_dir:
    :param model_dir:
    :return:
    """
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(model_dir, 'rb') as f:
            train_writer = tf.compat.v1.summary.FileWriter(log_dir)
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
            train_writer.add_graph(sess.graph)
            print(f"$ graph is write to {log_dir}")
