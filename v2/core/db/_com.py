import cv2
import numpy as np
from pathlib import Path
from uuid import uuid1
from typing import List, Generator

# model
from v2.contrib.images import Image

# Exception
from v2.contrib.images.exceptions import *
from .exceptions import *


class Identity:
    PREFIX_MASK = 'mask'
    PREFIX_NORMAL = 'normal'
    PREFIX_ALL = 'all'

    def __init__(self, root_path: Path, uu_id: str, *args, **kwargs):
        if not root_path.is_dir():
            raise InvalidDatabasePathError(f"the path {str(root_path)} is not a directory")
        root_path.mkdir(parents=True, exist_ok=True)
        self._uu_id = uu_id
        self._root_path = root_path
        self._images_path = self._root_path.joinpath("images")
        self._images_path.mkdir(parents=True, exist_ok=True)
        em_path = self._root_path.joinpath("embeddings")
        em_path.mkdir(parents=True, exist_ok=True)
        self._mask_npy_file = em_path.joinpath("mask.npy")
        self._normal_npy_file = em_path.joinpath("normal.npy")

        if kwargs["is_new"]:
            pass
        else:
            self._images = []

        super(Identity, self).__init__(*args, **kwargs)

    @property
    def uu_id(self) -> str:
        return self._uu_id

    def load_embedding(self, prefix: str = "normal") -> np.ndarray:
        if prefix == Identity.PREFIX_NORMAL:
            try:
                return np.load(str(self._normal_npy_file))
            except FileNotFoundError:
                raise NpyFileNotFoundError(f"npy file not founded at path {str(self._normal_npy_file)}")
        elif prefix == Identity.PREFIX_MASK:
            try:
                return np.load(str(self._mask_npy_file))
            except FileNotFoundError:
                raise NpyFileNotFoundError(f"npy file not founded at path {str(self._mask_npy_file)}")
        else:
            raise InvalidPrefixError("passing invalid prefix")

    def write_embeddings(self, mat: np.ndarray, prefix: str = 'normal'):
        if prefix == Identity.PREFIX_NORMAL:
            np.save(str(self._normal_npy_file), mat)
        elif prefix == Identity.PREFIX_MASK:
            np.save(str(self._mask_npy_file), mat)
        else:
            raise InvalidPrefixError("passing invalid prefix")

    def __generate_image_file_name(self, prefix: str = 'normal') -> Path:
        __n_filename = uuid1().hex
        if prefix == Identity.PREFIX_NORMAL or prefix == Identity.PREFIX_MASK:
            return self._images_path.joinpath(f"{prefix}_{__n_filename}.jpg")
        else:
            raise InvalidPrefixError("passing invalid prefix")

    def write_images(self, images: List[Image], prefix: str = 'normal') -> None:
        """
        write images into specific directory
        :param images: List[Image]
        :param prefix: type of image
        :return:
        """

        for im in images:
            __path = self.__generate_image_file_name(prefix)
            cv2.imwrite(str(__path), im.get_pixel)

    def load_images(self, prefix: str = 'normal', in_memory: bool = False) -> Generator:
        """
        load images
        :param in_memory:
        :param prefix: type of images
        :return: List[Image]
        """
        if prefix == Identity.PREFIX_NORMAL:
            for _p in self._images_path.glob("normal_*.jpg"):
                yield Image(file_path=_p, im=None, in_memory=in_memory)
        elif prefix == Identity.PREFIX_MASK:
            for _p in self._images_path.glob("mask_*.jpg"):
                yield Image(file_path=_p, im=None, in_memory=in_memory)
        elif prefix == Identity.PREFIX_ALL:
            for _p in self._images_path.glob("*.jpg"):
                yield Image(file_path=_p, im=None, in_memory=in_memory)
        else:
            raise InvalidPrefixError("passing invalid prefix")


class BasicDatabase:
    def __init__(self, *args, **kwargs):
        super(BasicDatabase, self).__init__(*args, **kwargs)

    def _parse(self, *args, **kwargs):
        raise NotImplementedError

    def get_embedded(self, *args, **kwargs):
        raise NotImplementedError

    def __initial_db(self):
        raise NotImplementedError

    @property
    def is_stable(self) -> bool:
        raise NotImplementedError

    def search(self, uu_id: str):
        raise NotImplementedError
