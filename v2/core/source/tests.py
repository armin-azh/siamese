from unittest import TestCase
import numpy as np
from ._image import SourceImage

# exception
from v2.contrib.images.exceptions import *


class SourceImageTestCase(TestCase):
    def test_create_source_image(self):
        im = np.random.random((512, 512))
        obj = SourceImage(im)

        self.assertEqual(obj.size, im.shape)
        self.assertEqual(obj.d_type, im.dtype)
