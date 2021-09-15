import numpy as np


class Policy:
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        raise NotImplementedError


class FaPolicyV1(Policy):
    def __init__(self, max_life_time: float, max_confidence: int, *args, **kwargs):
        super(FaPolicyV1, self).__init__(*args, **kwargs)
        self._max_life = max_life_time
        self._max_conf = max_confidence

        self._un_trackers = []
        self._trackers = []
    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        pass
