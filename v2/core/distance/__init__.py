from .base import BaseDistance
from ._dist import (CosineDistance,
                    SklearnCosineDistance,
                    EuclideanDistance,
                    SklearnEuclideanDistance,
                    SiPyMahalanobisDistance512)

Distance = BaseDistance
CosineDistanceV1 = CosineDistance
CosineDistanceV2 = SklearnCosineDistance
EuclideanDistanceV1 = EuclideanDistance
EuclideanDistanceV2 = SklearnEuclideanDistance
MahalanobisDistanceV1 = SiPyMahalanobisDistance512
