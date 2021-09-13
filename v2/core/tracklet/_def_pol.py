import numpy as np


class Policy:
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        raise NotImplementedError


class FaPolicyV1(Policy):
    def __init__(self, *args, **kwargs):
        super(FaPolicyV1, self).__init__(*args, **kwargs)

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        pass
