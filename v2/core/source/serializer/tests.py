from unittest import TestCase

# models
from ._field import *

# exceptions
from .exceptions import *


class BaseFieldTestCase(TestCase):

    def test_create_new_field_with_name(self):
        name = "charfield"
        dtype = str
        req = False
        BaseField(name=name, dtype=dtype, required=req)
