from unittest import TestCase
import numpy as np
from pathlib import Path

# models
from settings import BASE_DIR
from ._base import *

# exceptions
from .exceptions import *


class BaseImageTestCase(TestCase):
    def test_create_base_image(self):
        tm = np.random.random((560, 60))
        obj = BaseImage(im=tm)
        self.assertEqual(obj.size, tm.shape)
        self.assertEqual(obj.d_type, tm.dtype)


class ImageTestCase(TestCase):
    def setUp(self) -> None:
        self._im_path = Path(BASE_DIR).joinpath("v2/core/network/data/celeb.jpg")
        self._no_im_path = Path(BASE_DIR).joinpath("v2/core/network/data/celebc.jpg")
        self._im_in = cv2.imread(str(self._im_path))
        self._im = np.random.random((512, 512, 3))

    def test_create_image_with_invalid_path(self):
        with self.assertRaises(FileExistsError):
            Image(self._no_im_path, None)

    def test_create_image_with_no_path_no_image(self):
        with self.assertRaises(UndefinedSate):
            Image(None, None)

    def test_create_image_without_path_with_image(self):
        image = Image(file_path=None, im=self._im)
        self.assertTrue(image.is_memory)

        with self.assertRaises(UndefinedSate):
            image.memory_switch()

        self.assertTrue(image.is_memory)

    def test_create_image_with_path_without_image_in_memory_on(self):
        image = Image(file_path=self._im_path, im=None)
        self.assertEqual(self._im_in.shape, image.size)
        self.assertEqual(image.d_type, self._im_in.dtype)

    def test_create_image_with_path_without_image_in_memory_off(self):
        image = Image(file_path=self._im_path, im=None, in_memory=False)

        with self.assertRaises(NoImageLoadedError):
            _ = image.d_type

        with self.assertRaises(NoImageLoadedError):
            _ = image.size

        with self.assertRaises(NoImageLoadedError):
            _ = image.get_pixel

        image.memory_switch()

        self.assertEqual(self._im_in.shape, image.size)
        self.assertEqual(image.d_type, self._im_in.dtype)

        a = image.get_pixel
        self.assertIsNotNone(a)

        image.memory_switch()

        with self.assertRaises(NoImageLoadedError):
            _ = image.d_type

        with self.assertRaises(NoImageLoadedError):
            _ = image.size

        with self.assertRaises(NoImageLoadedError):
            _ = image.get_pixel
