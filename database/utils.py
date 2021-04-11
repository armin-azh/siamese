from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import ntpath
import logging


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


def get_logger(logger_name: str, log_dir: str, log_name: str):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)')
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.ERROR)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return logger
