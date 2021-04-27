from abc import ABC

import cv2
from settings import MOTION_CONF
import numpy as np
from recognition.utils import Counter
from skimage.metrics import structural_similarity as ssim


class BaseMotionDetection:
    def __init__(self):
        pass

    def has_motion(self, im_frame):
        return self._run(im_frame)

    def _run(self, im_frame):
        """
        in this method, we implement motion detection idea
        :param im_frame:
        :return: bool
        """
        raise NotImplementedError("this method should be implement")


class BSMotionDetection(BaseMotionDetection, ABC):
    """
    Background subtraction motion detection
    """

    def __init__(self):
        super(BSMotionDetection, self).__init__()
        self._counter = Counter(int(MOTION_CONF.get("bg_change_step")))
        self._backgournd = None

    def _run(self, im_frame):
        im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
        im_frame = cv2.GaussianBlur(im_frame, (21, 21), 0)

        if self._backgournd is None or self._counter() == 0:
            self._counter.next()
            self._backgournd = im_frame
            return None

        self._counter.next()

        delta_frame = cv2.absdiff(self._backgournd, im_frame)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 2:
            cnts = cnts[0]

        elif len(cnts) == 3:
            cnts = cnts[1]

        maximum = -1
        box = None
        for c in cnts:
            # if cv2.contourArea(c) < 100:
            #     continue

            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(im_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > maximum:
                box = (x, y, w, h)

        if box is not None:
            return box
        else:
            return None


class SsimMotionDetection(BaseMotionDetection):
    """
    SSIM movment detection
    """

    def __init__(self, thresh=0.9):
        super(SsimMotionDetection, self).__init__()
        self._counter = Counter(int(MOTION_CONF.get("ssim change")))
        self._backgournd = None
        self._thresh = thresh

    def _run(self, im_frame):
        im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)

        if self._backgournd is None or self._counter() == 0:
            self._counter.next()
            self._backgournd = im_frame
            return None

        self._counter.next()

        ssim_mean = ssim(self._backgournd, im_frame)
        if ssim_mean > self._thresh:
            return None

        return im_frame

# if __name__ == "__main__":
#     motion = BSMotionDetection()
#
#     cap = cv2.VideoCapture(0)
#
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         has_motion = motion.has_motion(frame)
#
#         print(has_motion)

