from mtcnn import MTCNN
import os
import configparser
from PIL import Image
import tensorflow as tf
import string
import random
import cv2
from functools import partial
import numpy as np
import time
from settings import BASE_DIR
from datetime import datetime


class FaceDetector:
    def __init__(self, sess, o_size: tuple = None):
        conf = configparser.ConfigParser()
        conf.read(os.path.join(BASE_DIR, "conf.ini"))
        image_conf = conf['Image']
        self._default = conf["Default"]
        self._detector_conf = conf['Detector']
        self._o_shape = o_size if o_size is not None else (int(image_conf.get('width')), int(image_conf.get('height')))
        thresholds = [float(self._detector_conf.get('step1_threshold')),
                      float(self._detector_conf.get('step2_threshold')),
                      float(self._detector_conf.get('step3_threshold'))]
        tf.compat.v1.keras.backend.set_session(sess)
        self.detector = MTCNN(steps_threshold=thresholds)

    @staticmethod
    def generate_name(name, prefix, im_format='jpg'):
        """
        generate file name
        :param name:
        :param prefix:
        :param im_format:
        :return: string
        """
        prefix = prefix + '.' + im_format
        return '_'.join([name, prefix])

    def extract_face(self, im, bbox):
        """
        this method will extract faces with giver bounding box
        :param im:
        :param bbox: tuple -> (x,y,w,h)
        :return: image array
        """
        x0, y0, w, h = bbox
        x1, y1 = abs(x0), abs(y0)
        x2, y2 = x1 + w, y1 + h
        im = im[y1:y2, x1:x2]
        im = Image.fromarray(im)
        im = im.resize(self._o_shape)
        return im

    def extract_faces(self, frame, area):
        frame = Image.fromarray(frame)
        frame = frame.convert('RGB')
        frame = np.asarray(frame)
        results = self.detector.detect_faces(frame)
        for res in results:
            _, _, w_, h_ = res.get('box')
            area_ = w_ * h_
            ratio = (float(area_) / area)
            if float(self._default.get('min_ratio')) < ratio < float(self._default.get('max_ratio')):
                yield self.extract_face(frame, res.get('box')), res.get('box')

    # def extract_faces(self, videos_file, postfix):
    #     """
    #     :param postfix:
    #     :param videos_file:
    #     :return:
    #     """
    #     base_save_path = os.path.join(BASE_DIR, self._detector_conf.get("default_save_path"))
    #
    #     prefix = ''.join(random.choice(string.ascii_lowercase) for i in range(12))
    #     name_fmt = partial(FaceDetector.generate_name, prefix=prefix)
    #
    #     subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    #
    #     base_save_path = os.path.join(base_save_path, subdir, postfix)
    #     if not os.path.exists(base_save_path):
    #         os.makedirs(base_save_path)
    #
    #     cnt_glob = 0
    #     f_cnt = 0
    #
    #     prev = time.time()
    #
    #     for f_path in videos_file:
    #
    #         cap = cv2.VideoCapture(f_path)
    #         cnt = 0
    #
    #         while cap.isOpened():
    #             delta_time = time.time() - prev
    #             res, frame = cap.read()
    #             if not res:
    #                 break
    #
    #             if delta_time > 1. / self._detector_conf.get('fps'):
    #                 prev = time.time()
    #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 image = Image.fromarray(frame)
    #                 image = image.convert('RGB')
    #                 im_pixels = np.asarray(image)
    #                 results = self.detector.detect_faces(im_pixels)
    #                 for ent in results:
    #                     tm_im = im_pixels
    #                     if ent['confidence'] > self._detector_conf.get('conf_thresh'):
    #                         tm_im = self.extract_face(tm_im, ent['box'])
    #                         tm_im.save(os.path.join(base_save_path, name_fmt(str(f_cnt))))
    #                         if cnt % self._detector_conf.get('dip_step') == 0 and cnt > 0:
    #                             print(f'$ {str(cnt)} faces has saved.')
    #                         cnt += 1
    #                         f_cnt += 1
    #
    #         cnt_glob += cnt
    #         print(f'$ totally {cnt} faces had been extracted from {f_path} file.')
    #         cap.release()
    #
    #     print(f'$ totally {cnt_glob} faces had been extracted from {str(len(videos_file))} files.')
