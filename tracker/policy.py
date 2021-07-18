from .counter import Counter
import time
import copy


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
        self._alias_name = None

    @property
    def counter(self) -> int:
        return self._counter()

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

    @property
    def alias_name(self):
        return self._alias_name

    @alias_name.setter
    def alias_name(self, n):
        self._alias_name = n

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

    def search_alias_name(self, name: str):
        res = None
        for idx, pol in enumerate(self._policy_list):
            if pol.alias_name == name:
                res = (idx, pol)
                break
        return res

    def _search(self, name: str):
        res = None
        for idx, pol in enumerate(self._policy_list):
            if pol.name == name:
                res = (idx, pol)
                break
        return res

    def modify(self):
        for pol in self._policy_list:
            if pol.status == Policy.STATUS_EXPIRED:
                tm = copy.deepcopy(pol)
                self._policy_list.remove(pol)
                yield tm

    def __call__(self, name: str, alias_name: str):

        res = self._search(name)

        if res is None:
            n_pol = Policy(name, self._max_life_time, self._max_conf)
            n_pol.alias_name = alias_name
            self._policy_list.append(n_pol)
            return n_pol, None
        else:
            _, pol = res
            pol()
            expired = None
            if pol.status == Policy.STATUS_EXPIRED:
                expired = copy.deepcopy(pol)
                self._policy_list.remove(pol)
                pol = Policy(name, self._max_life_time, self._max_conf)
                pol.alias_name = alias_name
                self._policy_list.append(pol)

            return pol, expired
