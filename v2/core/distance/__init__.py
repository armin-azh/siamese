from .base import BaseDistance
from ._dist import CosineDistance, SklearnCosineDistance, EuclideanDistance,SklearnEuclideanDistance

Distance = BaseDistance
CosineDistanceV1 = CosineDistance
CosineDistanceV2 = SklearnCosineDistance
EuclideanDistanceV1 = EuclideanDistance
EuclideanDistanceV2 = SklearnEuclideanDistance
