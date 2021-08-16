from typing import Tuple
from typing import Union

import cv2
import numpy as np
import tensorflow as tf

from ._basic import BasicService
from pathlib import Path

from v2.core.source import SourcePool
from v2.core.network import MultiCascadeFaceDetector
from v2.core.network import FaceNetModel
from v2.core.db import SimpleDatabase
from v2.core.distance import CosineDistanceV2, CosineDistanceV1


class EmbeddingService(BasicService):
    def __init__(self, name, log_path: Path, source_pool: SourcePool, face_detector: Union[MultiCascadeFaceDetector],
                 embedded: Union[FaceNetModel], database: Union[SimpleDatabase],
                 distance: Union[CosineDistanceV2, CosineDistanceV1], display=True,
                 *args, **kwargs):
        self._vision = source_pool
        self._f_d = face_detector
        self._embedded = embedded
        self._db = database
        self._dist = distance
        super(EmbeddingService, self).__init__(name=name, log_path=log_path, display=display, *args, **kwargs)


class RawVisualService(EmbeddingService):
    def __init__(self, name, log_path: Path, display=True, *args, **kwargs):
        super(RawVisualService, self).__init__(name=name, log_path=log_path, display=display, *args, **kwargs)

    def exec_(self, *args, **kwargs) -> None:

        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        msg = f"[Start] recognition server is now starting"
        self._file_logger.info(msg)
        if self._display:
            self._console_logger.success(msg)

        with tf.device('/device:gpu:0'):
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:

                    self._f_d.load_model(session=sess)

                    while True:
                        v_frame, v_id, v_timestamp = self._vision.next_stream()

                        if v_frame is None and v_id is None and v_timestamp is None:
                            continue

                        if cv2.waitKey(1) == ord("q"):
                            break

                        f_bound, f_landmarks = self._f_d.extract(im=v_frame)

                        print(f_bound.shape)
                        print(f_landmarks.shape)

                        window_name = f"{v_id[0:5]}..."
                        cv2.imshow(window_name, v_frame)
