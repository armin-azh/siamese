from argparse import ArgumentParser, Namespace
from pathlib import Path

from test.settings import datasets
from test.inspect import inspect_youtube


def main(arguments: Namespace):
    db_path = Path(arguments.input_dir)
    if arguments.db == "youtube":
        if arguments.inspect:
            inspect_youtube(db_path)
        elif arguments.process:
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--db', help="choose the dataset", type=str, default="youtube", choices=datasets)
    parser.add_argument("--input", help="input directory", type=str, default="G:\\Documents\\Project\\facerecognition"
                                                                             "\data\\dataset\\YouTubeFaces"
                                                                             "\\frame_images_DB", dest="input_dir")
    parser.add_argument("--inspect", help="inspect the dataset", action="store_true")
    parser.add_argument("--process", help="process the dataset", action="store_true")
    args = parser.parse_args()

    main(args)
