from typing import Tuple
from ._basic import BasicService
from pathlib import Path


class EmbeddingService(BasicService):
    def __init__(self, name, log_path: Path, *args, **kwargs):
        super(EmbeddingService, self).__init__(name=name, log_path=log_path, *args, **kwargs)
        self._vision = kwargs["source_pool"]
        self._f_d = kwargs["face_detector"]
        self._embedded = kwargs["embedded"]



