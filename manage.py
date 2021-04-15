import argparse
import configparser
import os
import pathlib
from recognition.recognition import face_recognition, face_recognition_on_keras
from recognition.utils import convert_computation_graph_to_keras_model
from database.component import inference_db
from settings import BASE_DIR


def main(args):
    if (args.realtime or args.video or args.cluster) and not args.keras:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', help="realtime recognition flag", action='store_true')
    parser.add_argument('--video', help="video recognition flag", action='store_true')
    parser.add_argument('--video_file', help='video filename for recognition', type=str, default="")
    parser.add_argument('--eval_method', help='evaluation method', choices=['cosine', 'svm', 'euclidean'],
                        default='cosine')
    parser.add_argument('--cluster', help="cluster video frame", action='store_true')
    parser.add_argument('--cluster_name', help="cluster name for saving images", type=str, default='')
    parser.add_argument('--save',help="save frame in a file",action='store_true')
    parser.add_argument('--keras',help="use keras model",action="store_true")
    parser.add_argument('--classifier', help='classifier filename', type=str, default='')
    parser.add_argument('--db_check', help='check gallery status', action='store_true')
    parser.add_argument('--db_inspect', help="inspect database status", action='store_true')
    parser.add_argument('--db_build_npy', help='build npy embedding', action='store_true')
    parser.add_argument('--cnv_to_keras', help='convert computation graph to keras', action='store_true')
    parser.add_argument('--cnv_to_lite', help='convert keras to lite', action='store_true')

    args = parser.parse_args()

    main(args)
