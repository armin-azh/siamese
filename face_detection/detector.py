from mtcnn import MTCNN
import os
import configparser
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from settings import BASE_DIR, DETECTOR_CONF, IMAGE_CONF, DEFAULT_CONF
from typing import List


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

    def extract_faces(self, frame, width, height):
        if self._model_type == self.DT_MTCNN:
            frame = Image.fromarray(frame)
            frame = frame.convert('RGB')
            frame = np.asarray(frame)
            results = self.detector.detect_faces(frame)
            for res in results:
                _, _, w_, h_ = res.get('box')
                area_ = w_ * h_
                ratio = (float(area_) / (width * height))
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
                    yield self.extract_face(frame, b_box), b_box


class AbstractFaceDetector:
    def __init__(self, *args, **kwargs):
        super(AbstractFaceDetector, self).__init__(*args, **kwargs)

    def find_keypoint(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract_faces(self, img: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def map_keypoint(self, keypoint: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MultiCascadeFaceDetector(AbstractFaceDetector):
    def __init__(self, sess, f_width: int, f_height: int, *args, **kwargs):
        self._threshold = [float(DETECTOR_CONF.get('step1_threshold')),
                           float(DETECTOR_CONF.get('step2_threshold')),
                           float(DETECTOR_CONF.get('step3_threshold'))]
        if sess is not None:
            tf.compat.v1.keras.backend.set_session(sess)

        self._detector = MTCNN(steps_threshold=self._threshold,
                               scale_factor=float(DETECTOR_CONF.get("scale_factor")),
                               min_face_size=int(DETECTOR_CONF.get("min_face_size")))
        self._total_nof_keypoint = 14
        self._output_shape = (int(IMAGE_CONF.get('width')), int(IMAGE_CONF.get('height')))
        self._output_channel = int(IMAGE_CONF.get("channel"))
        self._frame_width = f_width
        self._frame_height = f_height
        self._area = self._frame_height * self._frame_width
        self._min_ratio = float(DEFAULT_CONF.get("min_ratio"))
        self._max_ratio = float(DEFAULT_CONF.get("max_ratio"))

        super(MultiCascadeFaceDetector, self).__init__(*args, **kwargs)

    def set_frame_dim(self, dim: tuple) -> None:
        self._frame_width, self._frame_height = dim

    def find_keypoint(self, img: np.ndarray) -> np.ndarray:
        frame = Image.fromarray(img)
        frame = frame.convert('RGB')
        frame = np.asarray(frame)
        results = self._detector.detect_faces(frame)
        nof_faces = len(results)
        pairs = np.zeros((nof_faces, self._total_nof_keypoint), dtype=np.int32)

        for idx in range(nof_faces):
            p = results[idx]
            b_ = p.get("box")
            k = p.get("keypoints")

            pairs[idx, 0] = int(b_[0])
            pairs[idx, 1] = int(b_[1])
            pairs[idx, 2] = int(b_[2])
            pairs[idx, 3] = int(b_[3])

            k_nose = k.get("nose")
            pairs[idx, 4] = int(k_nose[0])
            pairs[idx, 5] = int(k_nose[1])

            k_r_eye = k.get("right_eye")
            pairs[idx, 6] = int(k_r_eye[0])
            pairs[idx, 7] = int(k_r_eye[1])

            k_l_eye = k.get("left_eye")
            pairs[idx, 8] = int(k_l_eye[0])
            pairs[idx, 9] = int(k_l_eye[1])

            k_r_mouth = k.get("mouth_right")
            pairs[idx, 10] = int(k_r_mouth[0])
            pairs[idx, 11] = int(k_r_mouth[1])

            k_l_mouth = k.get("mouth_left")
            pairs[idx, 12] = int(k_l_mouth[0])
            pairs[idx, 13] = int(k_l_mouth[1])

        return pairs

    def extract_faces(self, img: np.ndarray, keypoint: np.ndarray) -> np.ndarray:
        f_s = np.zeros((keypoint.shape[0], self._output_shape[0], self._output_shape[1], self._output_channel))

        boxes = self.get_bounding_box(keypoint)

        boxes = self.limit_face_area(boxes)

        idx_ = 0
        for box in boxes:
            x_, y_, w_, h_ = box
            x1, y1 = abs(x_), abs(y_)
            x2, y2 = x1 + w_, y1 + h_
            sub_img = img[y1:y2, x1:x2]
            sub_img = Image.fromarray(sub_img)
            sub_img = np.array(sub_img.resize(self._output_shape))
            f_s[idx_, :, :, :] = sub_img
            idx_ += 1

        return f_s

    def map_keypoint(self, keypoint: np.ndarray) -> np.ndarray:
        m_k = np.zeros_like(keypoint)

        origin = keypoint[:, :2]
        eye_ = self.get_eyes(keypoint)
        mouth_ = self.get_mouths(keypoint)
        nose_ = self.get_nose(keypoint)

        origin_ = np.concatenate([origin, origin], axis=1)
        eye_ = np.abs(np.subtract(eye_, origin_))
        mouth_ = np.abs(np.subtract(mouth_, origin_))
        nose_ = np.abs(np.subtract(nose_, origin))

        m_k[:, 0:4] = self.get_bounding_box(keypoint)
        m_k[:, 4:6] = nose_
        m_k[:, 6:10] = eye_
        m_k[:, 10:] = mouth_

        return m_k

    def limit_face_area(self, boxes: np.ndarray) -> np.ndarray:
        fa_width = boxes[..., 2]
        fa_height = boxes[..., 3]
        fa_area = np.multiply(fa_width, fa_height)
        ratio_ = fa_area / float(self._area)
        indices, *s = np.where((ratio_ > self._min_ratio) & (ratio_ < self._max_ratio))

        return boxes[indices, ...]

    def get_bounding_box(self, keypoint: np.ndarray) -> np.ndarray:
        return keypoint[:, 0:4]

    def get_nose(self, keypoint: np.ndarray) -> np.ndarray:
        return keypoint[:, 4:6]

    def get_eyes(self, keypoint: np.ndarray) -> np.ndarray:
        return keypoint[:, 6:10]

    def get_mouths(self, keypoint: np.ndarray) -> np.ndarray:
        return keypoint[:, 10:]
