import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from typing import Tuple


class MaskDetector:
    INPUT_SHAPE = None

    def __init__(self, score_threshold: float, input_shape: Tuple[int, int]):
        self.INPUT_SHAPE = input_shape
        self._score_threshold = score_threshold

    def single_class_non_max_suppression(self, bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
        if len(bboxes) == 0: return []

        conf_keep_idx = np.where(confidences > conf_thresh)[0]

        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]

        pick = []
        xmin = bboxes[:, 0]
        ymin = bboxes[:, 1]
        xmax = bboxes[:, 2]
        ymax = bboxes[:, 3]

        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(confidences)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break

            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)

        return conf_keep_idx[pick]

    def normalized(self, im_mask: np.ndarray) -> np.ndarray:

        im_mask /= 255

        return im_mask

    def predict(self, model, img: np.ndarray) -> np.ndarray:

        if img.shape[0] > 0:
            img = preprocess_input(img)
            _pred = model.predict(img)
            return _pred
        else:
            return np.empty((0, 1))

    def get_cropped_pics(self, img: np.ndarray, boxes: np.ndarray, offset_perc: int, cropping: str = '',
                         interpolation=cv2.INTER_LINEAR) -> np.ndarray:

        ori_height = img.shape[0]
        ori_width = img.shape[1]

        pics = []

        for box in boxes:

            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            if cropping == 'large':

                if (xmax - xmin) > (ymax - ymin):
                    ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                    ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

                elif (ymax - ymin) > (xmax - xmin):
                    xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                    xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

            elif cropping == 'small':

                if (xmax - xmin) > (ymax - ymin):
                    xmin = int((box[2] + box[0]) / 2 - (box[3] - box[1] + 1) / 2)
                    xmax = int((box[2] + box[0]) / 2 + (box[3] - box[1] + 1) / 2) - 1

                elif (ymax - ymin) > (xmax - xmin):
                    ymin = int((box[3] + box[1]) / 2 - (box[2] - box[0] + 1) / 2)
                    ymax = int((box[3] + box[1]) / 2 + (box[2] - box[0] + 1) / 2) - 1

            new_size = xmax - xmin

            xmin = xmin - int(new_size * offset_perc)
            ymin = ymin - int(new_size * offset_perc)

            xmax = xmax + int(new_size * offset_perc)
            ymax = ymax + int(new_size * offset_perc)

            if xmin < 0:
                xmax = xmax - xmin
                xmin = 0

            if xmax >= (ori_width - 1):
                xmin = (ori_width - 1) - (xmax - xmin)
                xmax = ori_width - 1

            if ymin < 0:
                ymax = ymax - ymin
                ymin = 0

            if ymax >= (ori_height - 1):
                ymin = (ori_height - 1) - (ymax - ymin)
                ymax = ori_height - 1

            if xmin >= 0 and ymin >= 0 and xmax < ori_width and ymax < ori_height:
                c_pic = img[ymin:ymax, xmin:xmax]
                if cropping == 'small' or cropping == 'large':
                    c_pic = cv2.resize(c_pic,
                                       (int(self.INPUT_SHAPE[0] * (1 + 2 * offset_perc)),
                                        int(self.INPUT_SHAPE[1] * (1 + 2 * offset_perc))),
                                       interpolation=interpolation)
                else:
                    if c_pic.shape[0] > c_pic.shape[1]:
                        c_pic = cv2.resize(c_pic,
                                           (int((self.INPUT_SHAPE[0] * c_pic.shape[0] / c_pic.shape[1]) * (
                                                   1 + 2 * offset_perc)),
                                            int(self.INPUT_SHAPE[1] * (1 + 2 * offset_perc))),
                                           interpolation=interpolation)
                    else:
                        c_pic = cv2.resize(c_pic, (int(self.INPUT_SHAPE[0] * (1 + 2 * offset_perc)),
                                                   int((self.INPUT_SHAPE[1] * c_pic.shape[1] / c_pic.shape[0]) * (
                                                           1 + 2 * offset_perc))), interpolation=interpolation)

                pics.append(c_pic)
            else:
                pics.append(np.empty(0))

        pics = np.array(pics)
        return pics
