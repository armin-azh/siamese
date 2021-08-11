from .base import BaseModel
from ._face_detector import FaceDetector
from ._recognizer import FaceNetModel
from ._hpe import HeadPoseEstimatorModel
from ._mask import MaskClassifierModel

BaseModel = BaseModel
MultiCascadeFaceDetector = FaceDetector
FaceNetModel = FaceNetModel
HPEModel = HeadPoseEstimatorModel
MaskModel = MaskClassifierModel
