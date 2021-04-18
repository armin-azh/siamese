from mtcnn import MTCNN
import os
import configparser
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from settings import BASE_DIR


class FaceDetector:
    DT_MTCNN = 'mtcnn'
    DT_RES10 = 'res10'

    def __init__(self, sess, o_size: tuple = None):
        conf = configparser.ConfigParser()
        conf.read(os.path.join(BASE_DIR, "conf.ini"))
        image_conf = conf['Image']
        self._default = conf["Default"]
        self._detector_conf = conf['Detector']
        self._image_conf = conf["Image"]
        self._model_conf = conf['Model']
        self._model_type = self._detector_conf['type']
        self._o_shape = o_size if o_size is not None else (int(image_conf.get('width')), int(image_conf.get('height')))
        if self._model_type == self.DT_MTCNN:
            thresholds = [float(self._detector_conf.get('step1_threshold')),
                          float(self._detector_conf.get('step2_threshold')),
                          float(self._detector_conf.get('step3_threshold'))]
            if sess is not None:
                tf.compat.v1.keras.backend.set_session(sess)
            self.detector = MTCNN(steps_threshold=thresholds,
                                  scale_factor=float(self._detector_conf.get("scale_factor")),
                                  min_face_size=int(self._detector_conf.get("min_face_size")))

        elif self._model_type == self.DT_RES10:
            self.detector = cv2.dnn.readNetFromCaffe(self._model_conf['res10_proto'], self._model_conf["res10_model"])

        else:
            raise ValueError("this model detector is not exists")

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
        if self._model_type == self.DT_MTCNN:
            x0, y0, w, h = bbox
            x1, y1 = abs(x0), abs(y0)
            x2, y2 = x1 + w, y1 + h
            im = im[y1:y2, x1:x2]
            im = Image.fromarray(im)
            im = im.resize(self._o_shape)
            return im
        elif self._model_type == self.DT_RES10:
            x0, y0, x1, y1 = bbox
            x0, y0 = abs(x0), abs(y0)
            x1, y1 = abs(x1), abs(y1)
            im = im[y0:y1, x0:x1]
            im = Image.fromarray(im)
            im = im.resize(self._o_shape)
            return im

    def extract_faces(self, frame, width,height):
        if self._model_type == self.DT_MTCNN:
            frame = Image.fromarray(frame)
            frame = frame.convert('RGB')
            frame = np.asarray(frame)
            results = self.detector.detect_faces(frame)
            for res in results:
                _, _, w_, h_ = res.get('box')
                area_ = w_ * h_
                ratio = (float(area_) / (width*height))
                if float(self._default.get('min_ratio')) < ratio < float(self._default.get('max_ratio')):
                    yield self.extract_face(frame, res.get('box')), res.get('box')

        elif self._model_type == self.DT_RES10:
            threshold = float(self._detector_conf.get('res10_threshold'))
            blob = cv2.dnn.blobFromImage(frame, 1.0, (int(self._image_conf["width"]), int(self._image_conf["height"])),
                                         (104.0, 177.0, 123.0), False, False)
            self.detector.setInput(blob)
            pred = self.detector.forward()

            for result in pred[0, 0, :, :]:
                confidence = result[2]
                if confidence > threshold:
                    x_left_bottom = int(result[3] * width)
                    y_left_bottom = int(result[4] * height)
                    x_right_top = int(result[5] * width)
                    y_right_top = int(result[6] * height)
                    b_box = (x_left_bottom, y_left_bottom, x_right_top, y_right_top)
                    yield self.extract_face(frame,b_box),b_box
