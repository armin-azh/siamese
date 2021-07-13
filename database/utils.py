from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import ntpath
import logging
from tabulate import tabulate
from uuid import uuid1


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


def tabulate_print(database_dic):
    """
    this function print dataset dictionary in tabular format
    :param database_dic:
    :return:
    """
    classes = ["class"]
    n_images = ["amount"]
    nps = ["npy"]
    for key, value in database_dic.items():
        classes.append(key)
        n_images.append(len(value[0]))
        nps.append("#" if value[1] else "x")

    print(tabulate({"class": classes, "amount": n_images, "npy": nps}))


def load_images(images):
    for im in images:
        yield im.read_image_file()


def generate_new_id():
    return uuid1().hex
