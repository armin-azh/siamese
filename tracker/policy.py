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
