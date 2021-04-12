import argparse
from recognition.recognition import face_recognition


def main(args):
    if args.realtime or args.video:
        face_recognition(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', help="realtime recognition flag", action='store_true')
    parser.add_argument('--video', help="video recognition flag", action='store_true')
    parser.add_argument('--video_file', help='video filename for recognition', type=str, default="")
    parser.add_argument('--eval_method', help='evaluation method', choices=['cosine', 'svm', 'euclidean'],
                        default='cosine')
    args = parser.parse_args()

    main(args)
