from typing import Tuple
from typing import Union, List
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ._basic import BasicService
from pathlib import Path
from shutil import move
import random

from v2.core.source import FileSourceModel
from v2.core.cluster import k_mean_clustering
from v2.contrib.images import Image
from v2.core.source import SourcePool
from v2.core.network import (MultiCascadeFaceDetector,
                             FaceNetModel,
                             HPEModel,
                             MaskModel)
from v2.core.db import SimpleDatabase
from v2.core.nomalizer import GrayScaleConvertor, SpaceConvertor, FaceNet160Cropper
from v2.core.distance import CosineDistanceV2, CosineDistanceV1
from v2.tools import draw_cure_face, Counter
from v2.core.db.exceptions import *

from .exceptions import *

from settings import PATH_CAP


class EmbeddingService(BasicService):
    def __init__(self, name, log_path: Path, source_pool: Union[SourcePool, List[FileSourceModel]],
                 face_detector: Union[MultiCascadeFaceDetector],
                 embedded: Union[FaceNetModel], database: Union[SimpleDatabase],
                 distance: Union[CosineDistanceV2, CosineDistanceV1, None], mask_detector: Union[MaskModel, None],
                 hpe: Union[HPEModel], display=True,
                 *args, **kwargs):
        self._vision = source_pool
        self._f_d = face_detector
        self._embedded = embedded
        self._db = database
        self._dist = distance
        self._hpe_model = hpe
        self._mask_d = mask_detector
        self._gray_conv = GrayScaleConvertor()

        super(EmbeddingService, self).__init__(name=name, log_path=log_path, display=display, *args, **kwargs)

    def _scale_factor(self, origin_shape: Tuple[int, int], conv_shape: Tuple[int, int]) -> Tuple[float, float]:
        return origin_shape[0] / conv_shape[0], origin_shape[1] / conv_shape[1]

    def _get_origin_box(self, scale_factor: Tuple[float, float], boxes: np.ndarray) -> np.ndarray:
        if boxes.shape[0] > 0:
            x_min = boxes[:, 0] * scale_factor[0]
            y_min = boxes[:, 1] * scale_factor[1]
            x_max = boxes[:, 2] * scale_factor[0]
            y_max = boxes[:, 3] * scale_factor[1]
            return np.concatenate(
                [x_min.reshape((-1, 1)), y_min.reshape((-1, 1)), x_max.reshape((-1, 1)), y_max.reshape((-1, 1))],
                axis=1)

        else:
            return np.empty((0, 4))


class ClusteringService(EmbeddingService):
    def __init__(self, name, log_path: Path, n_cluster: int, display=True, *args, **kwargs):
        self._space_normalizer = SpaceConvertor((640, 480))
        self._face_net_160_nomr = FaceNet160Cropper()
        self._n_cluster = n_cluster
        super(ClusteringService, self).__init__(name=name, log_path=log_path, mask_detector=None, distance=None,
                                                display=display, *args, **kwargs)

    def exec_(self, phase, *args, **kwargs) -> None:

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if phase not in ["mask", "normal"]:
            raise PassInvalidPhaseError("you passed invalid phase name!")

        msg = f"[Start] clustering server is now starting"
        self._file_logger.info(msg)
        if self._display:
            self._console_logger.success(msg)

        with tf.device('/device:gpu:0'):
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:

                    # face detector
                    self._f_d.load_model(session=sess)
                    msg = f"[OK] face detector model loaded"
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    # head pose estimator
                    self._hpe_model.load_model()
                    msg = f"[OK] head pose estimator loaded"
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    # load embedding network
                    self._embedded.load_model()
                    msg = f"[OK] embedding model loaded."
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    n_src = len(self._vision)

                    for src in self._vision:

                        msg = f"[Start] {src} is now reading."
                        self._file_logger.info(msg)
                        if self._display:
                            self._console_logger.success(msg)

                        new_identity_uuid = src.stem

                        try:
                            iden = self._db.add_new_identity(new_identity_uuid)
                            msg = f"[New] {new_identity_uuid} identity created."
                            self._file_logger.info(msg)
                            if self._display:
                                self._console_logger.success(msg)
                        except IdentityIsExistsError:
                            iden = self._db.get_identity(new_identity_uuid)
                            msg = f"[Exists] {new_identity_uuid} identity is Exists."
                            self._file_logger.info(msg)
                            if self._display:
                                self._console_logger.warn(msg)

                        cnt = Counter(limit=None)
                        cap = cv2.VideoCapture(str(src))
                        color_cropped = []
                        gray_cropped = []
                        while cap.isOpened():
                            cnt.next()
                            ret, o_frame = cap.read()
                            if ret is None or o_frame is None:
                                break
                            v_frame = self._space_normalizer.normalize(o_frame)

                            scale_ratio = self._scale_factor(origin_shape=o_frame.shape, conv_shape=v_frame.shape)

                            f_bound, f_landmarks = self._f_d.extract(im=v_frame)
                            origin_f_bound = self._get_origin_box(scale_ratio, f_bound)
                            origin_gray_one_ch_frame = self._gray_conv.normalize(o_frame.copy(), channel="one")
                            origin_gray_full_ch_frame = self._gray_conv.normalize(o_frame.copy(), channel="full")

                            head_scores = self._hpe_model.estimate_poses(sess, origin_gray_one_ch_frame,
                                                                         origin_f_bound.astype(np.int))
                            has_head, has_no_head = self._hpe_model.validate_angle(head_scores)

                            if has_no_head.shape[0]:
                                msg = f"[Drop] {head_scores.shape[0]} face have been dropped."
                                self._file_logger.info(msg)
                                if self._display:
                                    self._console_logger.warn(msg)

                            if has_head.shape[0]:
                                origin_f_bound = origin_f_bound[has_head, :]
                                gray_cropped.append(self._face_net_160_nomr.normalize(mat=origin_gray_full_ch_frame,
                                                                                      b_mat=origin_f_bound.astype(
                                                                                          np.int),
                                                                                      interpolation=cv2.INTER_LINEAR,
                                                                                      offset_per=0,
                                                                                      cropping="large")[0, :, :, :])
                                color_cropped.append(Image(im=self._face_net_160_nomr.normalize(mat=o_frame,
                                                                                                b_mat=origin_f_bound.astype(
                                                                                                    np.int),
                                                                                                interpolation=cv2.INTER_LINEAR,
                                                                                                offset_per=0,
                                                                                                cropping="large")[0, :,
                                                              :, :],
                                                           file_path=None))

                        cap.release()
                        gray_cropped = np.array(gray_cropped)
                        if gray_cropped.shape[0] < self._n_cluster:
                            msg = f"[Failed] number of extracted faced is less than the number of clusters"
                            self._file_logger.info(msg)
                            if self._display:
                                self._console_logger.dang(msg)
                        if gray_cropped.shape[0] > 0:
                            embs = self._embedded.get_embeddings(session=sess, input_im=gray_cropped)
                            _, centroids = k_mean_clustering(embeddings=embs, n_cluster=self._n_cluster)
                            iden.write_embeddings(centroids, prefix=phase)

                        iden.write_images(color_cropped, prefix=phase)
                        if gray_cropped.shape[0] < self._n_cluster:
                            msg = f"[Finish] {new_identity_uuid} clustering finished."
                            self._file_logger.info(msg)
                            if self._display:
                                self._console_logger.dang(msg)

                        origin_src = Path(src)
                        suf = random.randint(0, 1000)
                        destination = PATH_CAP.joinpath(origin_src.stem + str(suf) + "_done" + origin_src.suffix)
                        move(str(src), destination)

                        msg = f"[Move] {origin_src.stem + origin_src.suffix} moved to {str(destination)}"
                        self._file_logger.info(msg)
                        if self._display:
                            self._console_logger.success(msg)

                    msg = f"[Done] Clustering service job finished."
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)


class RawVisualService(EmbeddingService):
    COLOR_DEFAULT = (0, 0, 255)
    COLOR_DODGER_BLUE = (255, 144, 30)

    def __init__(self, name, log_path: Path, display=True, *args, **kwargs):
        super(RawVisualService, self).__init__(name=name, log_path=log_path, display=display, *args, **kwargs)
        self._normal_em, self._normal_lb, self._normal_en, self._mask_em, self._mask_lb, self._mask_en = self._db.get_embedded()

    def _format_db(self):
        msg = f"$ [DB] Normal Embeddings: {self._normal_em.shape[0]}, Mask Embeddings: {self._mask_em.shape[0]}"
        self._file_logger.info(msg)
        if self._display:
            self._console_logger.success(msg)

    def _draw(self, mat: np.ndarray, box_mat: np.ndarray, label_mat: Union[np.ndarray, None] = None) -> np.ndarray:

        for _b in box_mat[:, :4]:
            _x_min, _y_min, _x_max, _y_max = _b
            mat = draw_cure_face(mat, (int(_x_min), int(_y_min)), (int(_x_max), int(_y_max)), 5, 10,
                                 self.COLOR_DODGER_BLUE, 2)

        return mat

    def exec_(self, *args, **kwargs) -> None:

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        msg = f"[Start] recognition server is now starting"
        self._file_logger.info(msg)
        if self._display:
            self._console_logger.success(msg)

        self._format_db()

        with tf.device('/device:gpu:0'):
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:

                    # face detector
                    self._f_d.load_model(session=sess)
                    msg = f"[OK] face detector model loaded"
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    # head pose estimator
                    self._hpe_model.load_model()
                    msg = f"[OK] head pose estimator loaded"
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    # mask detector
                    self._mask_d.load_model(session=sess)
                    msg = f"[OK] mask classifier loaded."
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    # load embedding network
                    self._embedded.load_model()
                    msg = f"[OK] embedding model loaded."
                    self._file_logger.info(msg)
                    if self._display:
                        self._console_logger.success(msg)

                    while True:
                        o_frame, v_frame, v_id, v_timestamp = self._vision.next_stream()

                        if v_frame is None and v_id is None and v_timestamp is None:
                            continue

                        scale_ratio = self._scale_factor(origin_shape=o_frame.shape, conv_shape=v_frame.shape)

                        if cv2.waitKey(1) == ord("q"):
                            break

                        f_bound, f_landmarks = self._f_d.extract(im=v_frame)
                        origin_f_bound = self._get_origin_box(scale_ratio, f_bound)
                        origin_gray_one_ch_frame = self._gray_conv.normalize(o_frame.copy(), channel="one")
                        origin_gray_full_ch_frame = self._gray_conv.normalize(o_frame.copy(), channel="full")

                        head_scores = self._hpe_model.estimate_poses(sess, origin_gray_one_ch_frame,
                                                                     origin_f_bound.astype(np.int))
                        has_head, has_no_head = self._hpe_model.validate_angle(head_scores)

                        if has_no_head.shape[0]:
                            msg = f"[Drop] {head_scores.shape[0]} face have been dropped."
                            self._file_logger.info(msg)
                            if self._display:
                                self._console_logger.warn(msg)

                        if has_head.shape[0] > 0:
                            mask_scores = self._mask_d.predict(origin_gray_full_ch_frame,
                                                               origin_f_bound[has_head, ...].astype(np.int))
                            has_mask, has_no_mask = self._mask_d.validate_mask(mask_scores)

                            print(has_mask)
                            print(has_no_mask)

                        # display
                        display_frame = self._draw(o_frame, origin_f_bound, None)
                        window_name = f"{v_id[0:5]}..."
                        cv2.imshow(window_name, display_frame)
