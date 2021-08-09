from unittest import TestCase
import numpy as np
import cv2
from pathlib import Path

# models
from ._base import BaseNormalizer
from ._norm import FaceNetNormalizer, HeadPoseEstimatorNormalizer
from ._conv import GrayScaleConvertor
from settings import BASE_DIR
from settings import HPE_CONF

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
        size = (0, 160, 160, 3)
        mat = np.empty(size)
        norm = FaceNetNormalizer()
        ans = norm.normalize(mat)
        self.assertEqual(ans.shape, size)


class GrayScaleConvertorTestCase(TestCase):
    def test_run_one_channel(self):
        cvt = GrayScaleConvertor()
        im = cv2.imread("G:\\Documents\\Project\\facerecognition\\v2\\core\\network\\data\\celeb.jpg")
        shape = im.shape[:2]
        im = cvt.normalize(im, channel="one")
        self.assertEqual(im.shape, shape)

    def test_run_full_channel(self):
        cvt = GrayScaleConvertor()
        im = cv2.imread("G:\\Documents\\Project\\facerecognition\\v2\\core\\network\\data\\celeb.jpg")
        shape = im.shape
        im = cvt.normalize(im, channel="full")
        self.assertEqual(im.shape, shape)


class HeadPoseEstimatorNormalizerTestCase(TestCase):
    def setUp(self) -> None:
        self._im_path = Path(BASE_DIR).joinpath("v2/core/network/data/celeb.jpg")
        self._im = cv2.imread(str(self._im_path))
        self._img_norm = (float(HPE_CONF.get("im_norm_mean")), float(HPE_CONF.get("im_norm_var")))

    def test_normalize_with_true_inputs(self):
        norm = HeadPoseEstimatorNormalizer()
        bounding_boxes = np.array([[54, 78, 75, 98], [12, 45, 32, 65]])

        ans = norm.normalize(mat=cv2.cvtColor(self._im, cv2.COLOR_BGR2GRAY),
                             b_mat=bounding_boxes,
                             offset_per=0,
                             cropping="large",
                             hpe_im_norm=self._img_norm,
                             interpolation=cv2.INTER_LINEAR)
        self.assertEqual(ans.shape, (2, 64, 64, 1))

    def test_normalize_with_invalid_input_image(self):
        norm = HeadPoseEstimatorNormalizer()
        bounding_boxes = np.array([[54, 78, 75, 98], [12, 45, 32, 65]])
        with self.assertRaises(InCompatibleDimError):
            _ = norm.normalize(mat=self._im,
                               b_mat=bounding_boxes,
                               offset_per=0,
                               cropping="large",
                               hpe_im_norm=self._img_norm,
                               interpolation=cv2.INTER_LINEAR)

    def test_normalize_with_no_passing_bounding_box(self):
        norm = HeadPoseEstimatorNormalizer()
        with self.assertRaises(NoPassingArgumentError):
            _ = norm.normalize(mat=self._im,
                               offset_per=0,
                               cropping="large",
                               hpe_im_norm=self._img_norm,
                               interpolation=cv2.INTER_LINEAR)

    def test_normalize_with_empty_bounding_box(self):
        norm = HeadPoseEstimatorNormalizer()
        bounding_boxes = np.empty((0, 4))

        ans = norm.normalize(mat=cv2.cvtColor(self._im, cv2.COLOR_BGR2GRAY),
                             b_mat=bounding_boxes,
                             offset_per=0,
                             cropping="large",
                             hpe_im_norm=self._img_norm,
                             interpolation=cv2.INTER_LINEAR)
        self.assertEqual(ans.shape, (0, 64, 64, 1))
