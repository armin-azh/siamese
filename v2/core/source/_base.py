import numpy as np
from uuid import uuid1
from typing import Tuple
from collections import deque
from threading import Thread
from pathlib import Path
from datetime import datetime

from v2.tools.logger import FileLogger, ConsoleLogger


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
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self._log_path = logg_path.joinpath(f"camera/{self._id}/{_cu}.log")
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._file_logger = FileLogger(self._log_path, self._id)
        self._console_logger = ConsoleLogger()

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
