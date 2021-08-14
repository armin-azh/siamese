from unittest import TestCase
import numpy as np
from v2.core.source._image import SourceImage
from .serializer._field import BaseField

# exception
from v2.contrib.images.exceptions import *


class SourceImageTestCase(TestCase):
    def test_create_source_image(self):
        im = np.random.random((512, 512))
        obj = SourceImage(im)

        self.assertEqual(obj.size, im.shape)
        self.assertEqual(obj.d_type, im.dtype)


class BaseFieldTestCase(TestCase):

    def test_create_new_field_with_name(self):
        name = "charfield"
        dtype = str
        req = False
        field = BaseField(name=name, dtype=dtype, required=req)

        with self.assertRaises(NotImplementedError):
            field.validate()

        with self.assertRaises(NotImplementedError):
            field.cleaned_date

        with self.assertRaises(NotImplementedError):
            field("")

