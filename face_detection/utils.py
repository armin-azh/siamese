import pathlib
from itertools import chain


def parse_file(base_path: str, f_format: list):
    base = pathlib.Path(base_path)
    if base.exists():
        for f in chain(f_format):
            for p in base.rglob('*.' + f):
                yield str(p)
    else:
        return None


# if __name__ == "__main__":
#     print(list(parse_file("G:\\Documents\\Project\\siamese\\data\\train\set_2\\train", ['jpg', 'JPG'])))
