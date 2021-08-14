import numpy as np
from uuid import uuid1
from typing import Tuple
from collections import deque
from threading import Thread
from pathlib import Path


class BaseSource:
    def __init__(self, uuid: str, src: str, output_size: Tuple[int, int], src_type: str, queue_size: int,
                 logg_path: Path, display: bool = False):
        self._id = uuid if uuid is not None else uuid1().hex
        self._src = src
        self._output_size = output_size
        self._src_type = src_type
        self._queue_size = queue_size
        self._frame_dequeue = deque(maxlen=self._queue_size)
        self._thread = Thread(target=self._get_frame, args=())
        self._display_log = display

    def _get_frame(self):
        pass

    def _run_thread(self):
        self._thread.daemon = True
        self._thread.start()

    def reset(self):
        pass

    def test(self):
        pass

    def release(self):
        pass

    def last_modified_time(self):
        pass
