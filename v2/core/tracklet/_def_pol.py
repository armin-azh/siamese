from typing import Union
import numpy as np
from v2.tools import Counter
from ._def_tracker import TrackerContainer


class Policy:
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def do(self, trk_ids: np.ndarray, *args, **kwargs):
        raise NotImplementedError


class FaPolicyV1(Policy):
    CONFIRMED = 'confirmed'
    NOT_CONFIRMED = 'not_confirmed'

    def __init__(self, max_life_time: float, max_confidence_rec: int, max_confidence_un_rec: int, *args, **kwargs):
        super(FaPolicyV1, self).__init__(*args, **kwargs)
        self._max_life = max_life_time
        self._max_conf_rec = max_confidence_rec
        self._max_conf_un_rec = max_confidence_un_rec

        self._trackers = []

    def _find(self, trk_id: int) -> Union[TrackerContainer, None]:
        _f = None
        for trk in self._trackers:
            if trk_id == trk.trk_id:
                _f = trk
                break
        return _f

    def do(self, trk_ids: np.ndarray, *args, **kwargs):

        confirmed_known_idx = []
        confirmed_unknown_idx = []
        not_confirmed_idx = []

        for idx, trk_id in trk_ids:
            _f = self._find(trk_id)

            if _f is None:
                not_confirmed_idx.append(idx)
                continue
            if _f.known_counter >= self._max_conf_rec:
                confirmed_known_idx.append(idx)

            elif _f.unknown_counter >= self._max_conf_un_rec:
                confirmed_unknown_idx.append(idx)

            else:
                not_confirmed_idx.append(idx)

        return np.array(confirmed_known_idx), np.array(confirmed_unknown_idx), np.array(not_confirmed_idx)
