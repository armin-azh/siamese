import numpy as np
from datetime import datetime

from v2.contrib.images import BaseImage


class SourceImage(BaseImage):
    def __init__(self, im: np.ndarray, *args, **kwargs):
        super(SourceImage, self).__init__(im, *args, **kwargs)
        self._time_stamp = datetime.now().timestamp()
