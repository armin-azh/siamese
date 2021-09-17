from typing import Union, Tuple, Dict, List
import copy
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

    KNOWN = None
    UNKNOWN = None

    def __init__(self, max_life_time: float, max_confidence_rec: int, max_confidence_un_rec: int, *args, **kwargs):
        super(FaPolicyV1, self).__init__(*args, **kwargs)
        self._max_life = max_life_time
        self._max_conf_rec = max_confidence_rec
        self._max_conf_un_rec = max_confidence_un_rec
        self.KNOWN = TrackerContainer.KNOWN
        self.UNKNOWN = TrackerContainer.UNKNOWN

        self._trackers = []

    def _find(self, trk_id: int) -> Union[TrackerContainer, None]:
        _f = None
        for trk in self._trackers:
            if trk_id == trk.trk_id:
                _f = trk
                break
        return _f

    def find(self, trk_id) -> Union[TrackerContainer, None]:
        return self._find(trk_id)

    def add(self, trk: TrackerContainer) -> None:
        self._trackers.append(trk)

    def do(self, trk_ids: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        confirmed_known_idx = []
        confirmed_unknown_idx = []
        not_confirmed_idx = []

        for idx, trk_id in enumerate(trk_ids):
            _f = self._find(trk_id)

            if _f is None:
                not_confirmed_idx.append(idx)
                continue

            _mvp = _f.most_valuable_id
            if (_mvp[0] is not None and _mvp[1] is not None) and (_mvp[1] >= self._max_conf_rec):
                _f(mat=None, identity=_mvp[0])
                confirmed_known_idx.append(idx)

            elif _f.unknown_counter >= self._max_conf_un_rec:
                _f(mat=None, identity=TrackerContainer.UNKNOWN)
                confirmed_unknown_idx.append(idx)

            else:
                not_confirmed_idx.append(idx)

        return np.array(confirmed_known_idx), np.array(confirmed_unknown_idx), np.array(not_confirmed_idx)

    def split(self, trk_ids: np.ndarray, conf_known: np.ndarray, conf_unknown: np.ndarray, not_conf: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        conf_kn, conf_un_kn, n_conf = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
        if len(conf_known) > 0:
            conf_kn = trk_ids[conf_known]
        if len(conf_unknown) > 0:
            conf_un_kn = trk_ids[conf_unknown]
        if len(not_conf) > 0:
            n_conf = trk_ids[not_conf]

        return conf_kn, conf_un_kn, n_conf

    def review(self) -> Dict[str, List[TrackerContainer]]:
        _ans = {"known_confirmed": [], "unknown_confirmed": [], "expired": []}

        for idx, trk in enumerate(self._trackers):
            _id, cnt = trk.most_valuable_id
            _add = False
            print(trk.send)
            if _id is not None and cnt is not None and (cnt >= self._max_conf_rec) and not trk.send:
                trk.sent()
                _add = True
                _ans["known_confirmed"].append(copy.deepcopy(trk))

            elif trk.unknown_counter >= self._max_conf_un_rec and not trk.send:
                trk.sent()
                _add = True
                _ans["unknown_confirmed"].append(copy.deepcopy(trk))

            if trk.delta > self._max_life and not _add:
                print("remove")
                if trk.send:
                    _ans["expired"].append(copy.deepcopy(trk))
                self._trackers.remove(trk)

        return _ans
