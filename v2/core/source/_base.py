import time

import numpy as np
from uuid import uuid1
from typing import Tuple
from collections import deque
from threading import Thread
from pathlib import Path
from datetime import datetime
import cv2
from heapq import heappop, heappush

from v2.tools.logger import FileLogger, ConsoleLogger
from v2.core.nomalizer import SpaceConvertor

from .exceptions import *

from v2.core.source._image import SourceImage


class BaseSource:
    def __init__(self, uuid: str, src, output_size: Tuple[int, int], src_type: str, queue_size: int,
                 logg_path: Path, display: bool = False, cvt=SpaceConvertor):
        self._id = uuid if uuid is not None else uuid1().hex
        self._src = src
        self._output_size = output_size
        self._org_size = None
        self._src_type = src_type
        self._queue_size = queue_size
        self._frame_dequeue = deque(maxlen=self._queue_size)
        self._display_log = display
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self._log_path = logg_path.joinpath(f"camera/{self._id}")
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_path.joinpath(f"{_cu}.log")
        self._file_logger = FileLogger(self._log_path, self._id)
        self._console_logger = ConsoleLogger()
        self._online = False
        self._cap = None
        self._load_network_stream()

        self._get_stream_thread = Thread(target=self._get_frame, args=())
        self._get_stream_thread.daemon = True
        self._get_stream_thread.start()
        msg = f"[Ok] Camera {self._id} is starting"
        self._file_logger.info(msg)
        if self._display_log:
            self._console_logger.success(msg)
        self._last_modified_time = self.__modify_date_time()
        self._convertor = cvt(resize=self._output_size)

    def _verify_network_stream(self):
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def _load_network_stream(self):
        def _load_network_stream_thread_f():
            if self._verify_network_stream():
                self._cap = cv2.VideoCapture(self._src)
                self._online = True

        self._load_stream_thread = Thread(target=_load_network_stream_thread_f, args=())
        self._load_stream_thread.daemon = True
        self._load_stream_thread.start()

    def _spin(self, secs):
        time.sleep(secs)

    def _get_frame(self):

        while True:
            try:
                if self._cap.isOpened() and self._online:
                    status, frame = self._cap.read()
                    if status:
                        self._frame_dequeue.append(SourceImage(im=frame))
                    else:
                        self._cap.release()
                        self._online = False
                else:
                    msg = f"[Reconnect] reconnect to the {self._src}"
                    self._file_logger.info(msg)
                    if self._display_log:
                        self._console_logger.warn(msg)
                    self._spin(2)
                self._spin(0.001)
            except AttributeError:
                pass

    def reset(self):
        pass

    def release(self):
        self._cap.release()

    @property
    def get_id(self) -> str:
        return self._id

    @property
    def last_modified_time(self) -> datetime:
        return self._last_modified_time

    def __modify_date_time(self) -> datetime:
        return datetime.now()

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def source_type(self) -> str:
        return self._src_type

    def stream(self):
        self._last_modified_time = self.__modify_date_time()
        if not self._online:
            self._spin(1)
            return None, None, None

        if self._frame_dequeue and self._online:
            frame = self._frame_dequeue[-1].get_pixel
            return self._convertor.normalize(mat=frame), frame, self._frame_dequeue[-1].timestamp
        else:
            return None, None, None

    def __gt__(self, other):
        return True if self._last_modified_time > other.last_modified_time else False


class SourcePool:
    def __init__(self, src_list):
        self._p_queue = []
        for s in src_list:
            heappush(self._p_queue, (s.last_modified_time, s))

    def next_stream(self):
        """
        :return: origin_matrix, matrix, id, timestamp
        """
        _, cap = heappop(self._p_queue)

        if cap.source_type == "file":
            origin_frame, frame, finished, timestamp = cap.stream()
            if finished and frame is None and origin_frame is None:
                return None, None, None, None
            else:
                heappush(self._p_queue, (cap.last_modified_time, cap))
                if not finished and frame is None:
                    return None, None, None, None
                else:
                    return origin_frame, frame, cap.get_id, timestamp

        elif cap.source_type == "protocol" or cap.source_type == "webCam":
            heappush(self._p_queue, (cap.last_modified_time, cap))
            origin_frame, frame, timestamp = cap.stream()
            if frame is None and timestamp is None and origin_frame is None:
                return None, None, None, None
            else:
                return origin_frame, frame, cap.get_id, timestamp
