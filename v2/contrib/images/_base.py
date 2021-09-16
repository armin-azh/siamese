import cv2
import numpy as np
from typing import Tuple, Union
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
    def __init__(self, file_path: Union[Path, None], im: Union[np.ndarray, None], in_memory: bool = True, *args,
                 **kwargs):
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


def add_margin(bbox: np.ndarray, margin: Tuple[int, int], im_shape: Tuple[int, int]) -> np.ndarray:
    """
    add margin to the giver bounding boxes
    :param bbox:
    :param margin:
    :param im_shape:
    :return:
    """
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    x1 = x1 - margin[0]
    y1 = y1 - margin[1]
    x2 = x2 + margin[0]
    y2 = y2 + margin[1]

    x1_ = np.stack([x1, np.array(len(x1) * [0])], axis=1).max(axis=1)
    y1_ = np.stack([y1, np.array(len(y1) * [0])], axis=1).max(axis=1)
    x2_ = np.stack([x2, np.array(len(x2) * [im_shape[1]])], axis=1).min(axis=1)
    y2_ = np.stack([y2, np.array(len(y2) * [im_shape[0]])], axis=1).min(axis=1)

    return np.stack([x1_, y1_, x2_, y2_], axis=1)
