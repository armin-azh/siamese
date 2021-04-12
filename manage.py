import argparse
from recognition.recognition import face_recognition
from database.component import inference_db


def main(args):
    if args.realtime or args.video or args.cluster:
        face_recognition(args)
    elif args.db_check or args.db_inspect:
        inference_db(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', help="realtime recognition flag", action='store_true')
    parser.add_argument('--video', help="video recognition flag", action='store_true')
    parser.add_argument('--video_file', help='video filename for recognition', type=str, default="")
    parser.add_argument('--eval_method', help='evaluation method', choices=['cosine', 'svm', 'euclidean'],
                        default='cosine')
    parser.add_argument('--cluster', help="cluster video frame", action='store_true')
    parser.add_argument('--cluster_name', help="cluster name for saving images", type=str, default='')
    parser.add_argument('--classifier', help='classifier filename', type=str, default='')
    parser.add_argument('--db_check', help='check gallery status', action='store_true')
    parser.add_argument('--db_inspect', help="inspect database status",action='store_true')

    args = parser.parse_args()

    main(args)
