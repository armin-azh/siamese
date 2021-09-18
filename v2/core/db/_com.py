import cv2
import numpy as np
from pathlib import Path
from uuid import uuid1
from typing import List, Generator, Union, Tuple
import json
from sklearn import preprocessing
from sklearn.preprocessing._label import LabelEncoder as Label_DT

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
        self._inv_mask_npy_file = em_path.joinpath("inv_mask.npy")
        self._inv_normal_npy_file = em_path.joinpath("inv_normal.npy")

        super(Identity, self).__init__(*args, **kwargs)

    @property
    def uu_id(self) -> str:
        return self._uu_id

    def load_inv_embedding(self, prefix: str = "normal") -> np.ndarray:
        if prefix == Identity.PREFIX_NORMAL:
            try:
                return np.load(str(self._inv_normal_npy_file))
            except FileNotFoundError:
                raise NpyFileNotFoundError(f"npy file not founded at path {str(self._normal_npy_file)}")
        elif prefix == Identity.PREFIX_MASK:
            try:
                return np.load(str(self._inv_mask_npy_file))
            except FileNotFoundError:
                raise NpyFileNotFoundError(f"npy file not founded at path {str(self._mask_npy_file)}")
        else:
            raise InvalidPrefixError("passing invalid prefix")

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

    def write_inv_embeddings(self, mat: np.ndarray, prefix: str = 'normal'):
        if prefix == Identity.PREFIX_NORMAL:
            np.save(str(self._inv_normal_npy_file), mat)
        elif prefix == Identity.PREFIX_MASK:
            np.save(str(self._inv_mask_npy_file), mat)
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


class Database(BasicDatabase):
    PREFIX = 'gallery'

    def __init__(self, db_path: Path, *args, **kwargs):
        if not db_path.is_dir():
            raise InvalidDatabasePathError(f"the path {str(db_path)} is not a directory")
        self._db_path = db_path
        self._db_path.mkdir(parents=True, exist_ok=True)
        super(Database, self).__init__(*args, **kwargs)

    def __initial_db(self):
        pass

    def __get_uu_ids(self) -> Generator:
        for _p in self._db_path.glob('*'):
            yield _p.stem

    def __search_uu_id(self, uu_id: str) -> Union[Path, None]:
        """
        check a specific uu_id is exists or not
        :param uu_id:
        :return:
        """
        _paths = list(self._db_path.glob(uu_id))
        return None if len(_paths) == 0 else _paths[0]

    def add_new_identity(self, uu_id: str):
        id_path = self.__search_uu_id(uu_id)
        if id_path is not None:
            raise IdentityIsExistsError(f"Identity wit ID {uu_id} is exists")
        n_id_path = self._db_path.joinpath(uu_id)
        n_id_path.mkdir(exist_ok=True)
        n_id = Identity(root_path=n_id_path, uu_id=uu_id)
        return n_id

    def get_identity(self, uu_id: str) -> Identity:
        id_path = self.__search_uu_id(uu_id)
        if id_path is None:
            raise IdentityIsExistsError(f"Identity with ID {uu_id} is Not exists")
        return Identity(root_path=id_path, uu_id=id_path.stem)

    def get_embedded(self, *args, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, Label_DT, np.ndarray, np.ndarray, Label_DT]:
        """

        :param args:
        :param kwargs:
        :return: normal_embeddings,normal_labels,normal_label_encoder,mask_embeddings,mask_labels,mask_label_encoder
        """
        mask_labels = []
        normal_labels = []

        masks_embed = []
        normals_embed = []

        normal_en_label = preprocessing.LabelEncoder()
        mask_en_label = preprocessing.LabelEncoder()

        for uu_id in self._db_path.glob("*"):
            id_container = self.get_identity(uu_id.stem)

            try:
                normal_embed = id_container.load_embedding(prefix="normal")
                normals_embed.append(normal_embed)
                normal_labels += [uu_id.stem] * normal_embed.shape[0]
            except NpyFileNotFoundError:
                pass

            try:
                mask_embed = id_container.load_embedding(prefix="mask")
                masks_embed.append(mask_embed)
                mask_labels += [uu_id.stem] * mask_embed.shape[0]
            except NpyFileNotFoundError:
                pass

        try:
            masks_embed = np.concatenate(masks_embed, axis=0)
        except ValueError:
            masks_embed = np.empty((0, 512))

        try:
            normals_embed = np.concatenate(normals_embed, axis=0)
        except ValueError:
            normals_embed = np.empty((0, 512))

        normal_en_label.fit(list(set(normal_labels)))
        mask_en_label.fit(list(set(mask_labels)))

        normal_labels = normal_en_label.transform(normal_labels)
        mask_labels = mask_en_label.transform(mask_labels)

        return normals_embed, normal_labels, normal_en_label, masks_embed, mask_labels, mask_en_label

    def get_inv_embedded(self, *args, **kwargs) -> Tuple[
        np.ndarray, np.ndarray, Label_DT, np.ndarray, np.ndarray, Label_DT]:
        """

        :param args:
        :param kwargs:
        :return: normal_embeddings,normal_labels,normal_label_encoder,mask_embeddings,mask_labels,mask_label_encoder
        """
        mask_labels = []
        normal_labels = []

        masks_inv_embed = []
        normals_inv_embed = []

        normal_en_label = preprocessing.LabelEncoder()
        mask_en_label = preprocessing.LabelEncoder()

        for uu_id in self._db_path.glob("*"):
            id_container = self.get_identity(uu_id.stem)

            try:
                normal_inv_embed = id_container.load_embedding(prefix="normal")
                normals_inv_embed.append(normal_inv_embed)
                normal_labels += [uu_id.stem] * normal_inv_embed.shape[0]
            except NpyFileNotFoundError:
                pass

            try:
                mask_inv_embed = id_container.load_embedding(prefix="mask")
                masks_inv_embed.append(mask_inv_embed)
                mask_labels += [uu_id.stem] * mask_inv_embed.shape[0]
            except NpyFileNotFoundError:
                pass

        try:
            masks_inv_embed = np.concatenate(masks_inv_embed, axis=0)
        except ValueError:
            masks_inv_embed = np.empty((0, 512))

        try:
            normals_inv_embed = np.concatenate(normals_inv_embed, axis=0)
        except ValueError:
            normals_inv_embed = np.empty((0, 512))

        normal_en_label.fit(list(set(normal_labels)))
        mask_en_label.fit(list(set(mask_labels)))

        normal_labels = normal_en_label.transform(normal_labels)
        mask_labels = mask_en_label.transform(mask_labels)

        return normals_inv_embed, normal_labels, normal_en_label, masks_inv_embed, mask_labels, mask_en_label
