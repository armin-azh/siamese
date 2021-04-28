from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import configparser
import pathlib

conf = configparser.ConfigParser()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
conf.read(os.path.join(BASE_DIR, "conf.ini"))

GALLERY_LOG_DIR = os.path.join(conf["Log"].get("gallery"), 'gallery') if conf["Log"].get(
    "gallery") is None else os.path.join(
    BASE_DIR, 'log/gallery')

MODEL_CONF = conf['Model']
GALLERY_CONF = conf['Gallery']
GALLERY_ROOT = pathlib.Path(BASE_DIR).joinpath(GALLERY_CONF.get("database_path"))
if not GALLERY_ROOT.exists():
    GALLERY_ROOT.mkdir()
DETECTOR_CONF = conf['Detector']
DEFAULT_CONF = conf['Default']
MOTION_CONF = conf["Motion"]

SETTINGS_HEADER = ["Default", "Image", "Gallery", "Detector", "Log", "Motion"]
