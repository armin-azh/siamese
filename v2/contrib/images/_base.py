import cv2
import numpy as np
from typing import Tuple
from pathlib import Path

# exceptions
from .exceptions import *


class BaseImage:
    def __init__(self, im: np.ndarray, *args, **kwargs):
        self._im = im
        self._origin_size = im.shape if im is not None else None
        self._d_type = im.dtype if im is not None else None

        super(BaseImage, self).__init__(*args, **kwargs)

    @property
    def get_pixel(self) -> np.ndarray:
        if self._im is None:
            raise NoImageLoadedError("No image had been loaded")
        return self._im.copy()

    @property
    def d_type(self):
        if self._d_type is None:
            raise NoImageLoadedError("No image had been loaded")
        return self._d_type

    @property
    def size(self) -> Tuple[int, int]:
        if self._origin_size is None:
            raise NoImageLoadedError("No image had been loaded")
        return self._origin_size


class Image(BaseImage):
    def __init__(self, file_path: Path, im: np.ndarray, in_memory: bool = True, *args, **kwargs):
        if file_path is None and im is None:
            raise UndefinedSate("you can`t create Image instance with passing None value for both file_path and im")
        if file_path is not None and (not file_path.exists() or not file_path.is_file()):
            raise FileExistsError("file is not exists")

        self._in_memory = True if file_path is None else in_memory
        self._im_path = file_path

        if self._im_path is not None and self._in_memory and im is None:
            super(Image, self).__init__(im=self.__read_image(), *args, **kwargs)
        else:
            super(Image, self).__init__(im=im, *args, **kwargs)

    @property
    def is_memory(self) -> bool:
        return self._in_memory

    def memory_switch(self, can=True) -> None:
        if self._im_path is None and can:
            raise UndefinedSate("Can`t memory off")

        self._in_memory = True if not self._in_memory else False
        if self._in_memory:
            self.__init(self.__read_image())
        else:
            self.__init(None)

    def __read_image(self) -> np.ndarray:
        return cv2.imread(str(self._im_path))

    def __init(self, im: np.ndarray) -> None:
        self._im = im
        self._origin_size = im.shape if im is not None else None
        self._d_type = im.dtype if im is not None else None
