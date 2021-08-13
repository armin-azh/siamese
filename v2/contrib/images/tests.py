from unittest import TestCase
import numpy as np

# models
from ._base import BaseImage


class BaseImageTestCase(TestCase):
    def test_create_base_image(self):
        tm = np.random.random((560, 60))
        obj = BaseImage(im=tm)
        self.assertEqual(obj.size, tm.shape)
        self.assertEqual(obj.d_type, tm.dtype)
