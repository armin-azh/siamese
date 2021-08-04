from unittest import TestCase
import unittest
import numpy as np

# models
from .base import BaseDistance
from ._dist import CosineDistance

# exception
from v2.core.exceptions import InCompatibleDimError


class BaseDistanceTestCase(TestCase):

    def test_check_name_of_the_class(self):
        new_class = BaseDistance()
        self.assertEqual(new_class.__class__.__name__, new_class.name)

    def test_satisfy_raise_not_implemented_error(self):
        new_class = BaseDistance()
        temp_arr = np.empty((12, 45))
        with self.assertRaises(NotImplementedError):
            new_class.satisfy(temp_arr)

    def test_calculate_distant_raise_not_implement_error(self):
        new_class = BaseDistance()
        temp_arr = np.empty((12, 45))
        with self.assertRaises(NotImplementedError):
            new_class.calculate_distant(temp_arr, temp_arr)

    def test_put_name_to_class(self):
        name = "mahalanobis/v1"
        new_class = BaseDistance(name)
        self.assertEqual(name, new_class.name)


class CosineDistanceTestCase(TestCase):
    def test_calculate_distant_with_compatible_dim(self):
        mat1 = np.random.random((12, 500))
        mat2 = np.random.random((60, 500))
        dis = CosineDistance(similarity_threshold=0.9)
        ans = dis.calculate_distant(mat1, mat2)
        self.assertEqual(ans.shape, (12, 60))

    def test_calculate_distant_with_incompatible_dim(self):
        mat1 = np.random.random((12, 100))
        mat2 = np.random.random((60, 500))
        dis = CosineDistance(similarity_threshold=0.9)
        with self.assertRaises(InCompatibleDimError):
            _ = dis.calculate_distant(mat1, mat2)

    def test_calculate_distant_with_incompatible_non_2dim(self):
        mat1 = np.random.random((12, 100))
        mat2 = np.random.random((60, 500, 2))
        dis = CosineDistance(similarity_threshold=0.9)
        with self.assertRaises(InCompatibleDimError):
            _ = dis.calculate_distant(mat1, mat2)

    def test_satisfy_with_compatible_dim(self):
        mat1 = np.random.random((12, 100))
        dis = CosineDistance(similarity_threshold=0.9)
        self.assertEqual(dis.satisfy(mat1).shape, (12,))

    def test_satisfy_with_incompatible_dim(self):
        mat1 = np.random.random((12, 100, 45))
        dis = CosineDistance(similarity_threshold=0.9)
        with self.assertRaises(InCompatibleDimError):
            _ = dis.satisfy(mat1)
