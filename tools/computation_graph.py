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


def computation_graph_inspect(model_dir,flags ):
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(model_dir, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

            if flags.get("node"):
                all_nodes = [n for n in tf.compat.v1.get_default_graph().as_graph_def().node]
                print(all_nodes)
            elif flags.get('ops'):
                all_ops = tf.compat.v1.get_default_graph().get_operations()
                print(all_ops)

            elif flags.get("vars"):
                all_vars = tf.compat.v1.global_variables()
                print(all_vars)

            elif flags.get("tensors"):
                all_tensors = [tensor for op in tf.compat.v1.get_default_graph().get_operations() for tensor in
                               op.values()]
                print(all_tensors)

            elif flags.get("placeholders"):
                all_placeholders = [placeholder for op in tf.compat.v1.get_default_graph().get_operations() if
                                    op.type == 'Placeholder' for placeholder in op.values()]
                print(all_placeholders)
