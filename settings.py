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

SUMMARY_LOG_DIR = pathlib.Path(BASE_DIR).joinpath("log/" + conf["Log"].get("summary"))

MODEL_CONF = conf['Model']
GALLERY_CONF = conf['Gallery']
GALLERY_ROOT = pathlib.Path(BASE_DIR).joinpath(GALLERY_CONF.get("database_path"))
if not GALLERY_ROOT.exists():
    GALLERY_ROOT.mkdir(parents=True)
DETECTOR_CONF = conf['Detector']
DEFAULT_CONF = conf['Default']
MOTION_CONF = conf["Motion"]
TRACKER_CONF = conf["Tracker"]
SOURCE_CONF = conf["Source"]
CAMERA_MODEL_CONF = conf["CameraModel"]
ZERO_MQ_CONF = conf["ZeroMQ"]
SERVER_CONF = conf["Server"]

SETTINGS_HEADER = ["Default", "Image", "Gallery", "Detector", "Log", "Motion", "Source", "ZeroMQ","Server"]

COLOR_SUCCESS = (0, 255, 0)
COLOR_DANG = (243, 32, 19)
COLOR_WARN = (255, 255, 0)
