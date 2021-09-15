import numpy as np
from v2.tools import Counter
from ._def_tracker import TrackerContainer


class Policy:
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        raise NotImplementedError


class FaPolicyV1(Policy):
    EXIST = 'exist'
    NOT_EXIST = 'not_exist'

    def __init__(self, max_life_time: float, max_confidence: int, *args, **kwargs):
        super(FaPolicyV1, self).__init__(*args, **kwargs)
        self._max_life = max_life_time
        self._max_conf = max_confidence

        self._trk_cnt = {}

        self._trackers = []

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        pass

    def _next_cnt(self, trk_id: int):
        try:
            self._trk_cnt[trk_id]()
        except KeyError:
            self._trk_cnt[trk_id] = Counter()
            self._trk_cnt[trk_id]()
