from .counter import Counter
import time


class Policy:
    STATUS_EXPIRED = 'expired'
    STATUS_NOT_CONF = 'not_conf'
    STATUS_CONF = 'conf'

    def __init__(self, name: str, max_life_time: float, max_conf: int):
        self._counter = Counter()
        self._counter.next()
        self._name = name
        self._max_life_time = max_life_time
        self._max_conf = max_conf
        self._last_modified = time.time()
        self._status = self.STATUS_NOT_CONF
        self._mark = False

    @property
    def mark(self) -> bool:
        return self._mark

    @mark.setter
    def mark(self, mk: bool):
        self._mark = mk

    @property
    def status(self):
        return self._status

    @property
    def name(self):
        return self._name

    def _modify(self):
        self._counter.next()

        val = self._validate()

        if val:
            self._last_modified = time.time()
        else:
            if self._status == self.STATUS_NOT_CONF:
                self._last_modified = time.time()

    def _validate(self):

        if self._status == self.STATUS_EXPIRED:
            return False

        else:
            delta = time.time() - self._last_modified
            if delta > self._max_life_time:
                self._status = self.STATUS_EXPIRED
                return False

            else:
                if self._counter() > self._max_conf:
                    self._status = self.STATUS_CONF
                    return True
                else:
                    self._status = self.STATUS_NOT_CONF
                    return False

    def __call__(self):
        self._modify()


class PolicyTracker:
    def __init__(self, max_life_time: float, max_conf: int):
        self._policy_list = []
        self._max_life_time = max_life_time
        self._max_conf = max_conf

    def __len__(self):
        return len(self._policy_list)

    def _search(self, name: str):
        res = None
        for idx, pol in enumerate(self._policy_list):
            if pol.name == name:
                res = (idx, pol)
                break
        return res

    def modify(self):

        for pol in self._policy_list:
            pol()
            if pol.status == Policy.STATUS_EXPIRED:
                self._policy_list.remove(pol)

    def __call__(self, name: str):

        self.modify()

        res = self._search(name)

        if res is None:
            n_pol = Policy(name, self._max_life_time, self._max_conf)
            self._policy_list.append(n_pol)
            return n_pol.status
        else:
            _, pol = res
            return pol.status
