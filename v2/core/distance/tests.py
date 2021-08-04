from unittest import TestCase
import unittest
import numpy as np

# models
from .base import BaseDistant


class BaseDistantTestCase(TestCase):

    def test_check_name_of_the_class(self):
        new_class = BaseDistant()
        self.assertEqual(new_class.__class__.__name__, new_class.name)

    def test_satisfy_raise_not_implemented_error(self):
        new_class = BaseDistant()
        temp_arr = np.empty((12, 45))
        with self.assertRaises(NotImplementedError):
            new_class.satisfy(temp_arr)

    def test_calculate_distant_raise_not_implement_error(self):
        new_class = BaseDistant()
        temp_arr = np.empty((12, 45))
        with self.assertRaises(NotImplementedError):
            new_class.calculate_distant(temp_arr, temp_arr)

    def test_put_name_to_class(self):
        name = "mahalanobis/v1"
        new_class = BaseDistant(name)
        self.assertEqual(name, new_class.name)
