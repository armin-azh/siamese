from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import time
import argparse
import face_detection
import numpy as np
import tensorflow as tf
from numpy import load
from mtcnn import MTCNN
from recognition import utils, distance
from sklearn import preprocessing
from face_detection.detector import FaceDetector


def cosine_realtime():
    prev = time.time()
    fd = FaceDetector()

    cap = cv2.VideoCapture(0)


