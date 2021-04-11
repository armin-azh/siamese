from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import ntpath


def write_json(data: dict, filename: str) -> None:
    """
    write data to jason file
    :param filename:
    :param data:
    :return: None
    """
    with open(filename, 'w') as output:
        json.dump(data, output)


def extract_filename(filename):
    """
    extract filename from full path
    :return:
    """
    file_path, _ = os.path.splitext(filename)
    base_dir, file_path = ntpath.split(file_path)
    return base_dir, file_path
