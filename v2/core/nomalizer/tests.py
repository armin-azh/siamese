from unittest import TestCase
import numpy as np

# models
from ._base import BaseNormalizer
from ._norm import FaceNetNormalizer

from v2.core.exceptions import *


class BaseNormalizerTestCase(TestCase):
    def test_create_base_normalizer_with_name(self):
        name = "z-score"
        bs = BaseNormalizer(name=name)
        self.assertEqual(bs.name, name)

    def test_create_base_normalizer_without_name(self):
        bs = BaseNormalizer()
        self.assertEqual(bs.name, bs.__class__.__name__)

    def test_call_normalize_method(self):
        bs = BaseNormalizer()
        mat = np.random.random((2, 2))
        with self.assertRaises(NotImplementedError):
            bs.normalize(mat)


class FaceNetNormalizerTestCase(TestCase):
    def test_run_normalizer_with_compatible_dim(self):
        size = (12, 160, 160, 3)
        mat = np.random.random(size)
        norm = FaceNetNormalizer()
        ans = norm.normalize(mat)
        self.assertEqual(ans.shape, size)

    def test_run_normalizer_with_incompatible_dim(self):
        size = (12, 160, 160)
        mat = np.random.random(size)
        norm = FaceNetNormalizer()

        with self.assertRaises(InCompatibleDimError):
            _ = norm.normalize(mat)

    def test_run_normalizer_empty(self):
        size = (0, 160, 160,3)
        mat = np.empty(size)
        norm = FaceNetNormalizer()
        ans = norm.normalize(mat)
        self.assertEqual(ans.shape, size)


