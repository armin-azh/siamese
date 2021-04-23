from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import configparser

conf = configparser.ConfigParser()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
conf.read(os.path.join(BASE_DIR, "conf.ini"))

GALLERY_LOG_DIR = os.path.join(conf["Log"].get("gallery"), 'gallery') if conf["Log"].get(
    "gallery") is None else os.path.join(
    BASE_DIR, 'log/gallery')

SETTINGS_HEADER = ["Default", "Image", "Gallery", "Detector", "Log"]
