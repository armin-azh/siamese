import argparse
import configparser
import os
import sys
from recognition.recognition import face_recognition, face_recognition_on_keras, test_recognition, cluster_faces
from recognition.recognition import face_recognition_kalman
from recognition.utils import convert_computation_graph_to_keras_model
from database.component import inference_db
from tools.system import system_status
from tools.download import download_models
from gui.main import *
from tools.computation_graph import computation_graph_inspect, pb_to_tensorboard
from tools.shadow import add_shadow
from settings import BASE_DIR


def main(args):
    if (args.realtime or args.video) and args.kalman_tracker:
        face_recognition_kalman(args)

    elif (args.realtime or args.video or args.cluster) and not args.keras:
        if args.cluster and args.cluster_bulk:
            cluster_faces(args)
        else:
            face_recognition(args)

    elif (args.realtime or args.video or args.cluster) and args.keras:
        face_recognition_on_keras(args)

    elif args.db_check or args.db_inspect or args.db_build_npy:
        inference_db(args)

    elif args.cnv_to_keras:
        conf = configparser.ConfigParser()
        conf.read(os.path.join(BASE_DIR, "conf.ini"))
        model_conf = conf["Model"]
        m_path = pathlib.Path(os.path.join(BASE_DIR, model_conf.get('facenet')))
        convert_computation_graph_to_keras_model(model_dir=m_path.parent, save_dir=m_path.parent.parent,
                                                 lite=args.cnv_to_lite)

    elif args.test:
        test_recognition(args)

    elif args.cg:
        if args.cg_inspect_ops or args.cg_inspect_nodes or args.cg_inspect_vars or args.cg_inspect_tensors or args.cg_inspect_placeholders:
            flags = {
                "node": args.cg_inspect_nodes,
                "ops": args.cg_inspect_ops,
                "vars": args.cg_inspect_vars,
                "tensors": args.cg_inspect_tensors,
                "placeholders": args.cg_inspect_placeholders
            }
            computation_graph_inspect(os.path.join(BASE_DIR, args.cg_cvt_graph_pb), flags)
        else:
            pb_to_tensorboard(os.path.join(BASE_DIR, args.cg_log_dir), os.path.join(BASE_DIR, args.cg_cvt_graph_pb))

    elif args.im_mani and (args.im_darker or args.im_brighter):
        add_shadow(args)

    elif args.gui:
        app = QApplication(sys.argv)
        window = MainWindow()
        sys.exit(app.exec_())

    elif args.sys:
        system_status(args)

    elif args.download_models:
        download_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', help="realtime recognition flag", action='store_true')
    parser.add_argument('--kalman_tracker', help='use kalman tracker', action='store_true')
    parser.add_argument('--video', help="video recognition flag", action='store_true')
    parser.add_argument('--video_file', help='video filename for recognition', type=str, default="")
    parser.add_argument('--eval_method', help='evaluation method', choices=['cosine', 'svm', 'euclidean'],
                        default='cosine')
    parser.add_argument('--cluster', help="cluster video frame", action='store_true')
    parser.add_argument('--cluster_name', help="cluster name for saving images", type=str, default='')
    parser.add_argument('--cluster_bulk', help="bulk clustering", action="store_true")
    parser.add_argument('--save', help="save frame in a file", action='store_true')
    parser.add_argument('--keras', help="use keras model", action="store_true")
    parser.add_argument('--classifier', help='classifier filename', type=str, default='')
    parser.add_argument('--db_check', help='check gallery status', action='store_true')
    parser.add_argument('--db_inspect', help="inspect database status", action='store_true')
    parser.add_argument('--db_build_npy', help='build npy embedding', action='store_true')
    parser.add_argument('--cnv_to_keras', help='convert computation graph to keras', action='store_true')
    parser.add_argument('--cnv_to_lite', help='convert keras to lite', action='store_true')
    parser.add_argument('--test', help="test a prob set on the gallery set", action='store_true')
    parser.add_argument('--test_dir', help="test directory", default='', type=str)
    parser.add_argument('--cg', help='computation graph flag', action='store_true')
    parser.add_argument('--cg_cvt_graph_pb', help='model pb to convert', default='', type=str)
    parser.add_argument('--cg_log_dir', help='log dir to store converted graph', default='', type=str)
    parser.add_argument('--cg_inspect_ops', help="operations in computation graph", action='store_true')
    parser.add_argument('--cg_inspect_nodes', help="nodes in computation graph", action='store_true')
    parser.add_argument('--cg_inspect_vars', help="variables in computation graph", action='store_true')
    parser.add_argument('--cg_inspect_tensors', help="tensors in computation graph", action='store_true')
    parser.add_argument('--cg_inspect_placeholders', help="placeholders in computation graph", action='store_true')
    parser.add_argument('--im_mani', help="image manipulation", action='store_true')
    parser.add_argument('--im_darker', help="darker images", action='store_true')
    parser.add_argument('--im_brighter', help="brighter images", action='store_true')
    parser.add_argument('--im_save', help="save manipulated images", default="", type=str)
    parser.add_argument('--gui', help="run gui", action='store_true')
    parser.add_argument('--sys', help="check system status and information", action="store_true")
    parser.add_argument("--download_models",help = "download model and save on local storage",action='store_true')

    args = parser.parse_args()

    main(args)
