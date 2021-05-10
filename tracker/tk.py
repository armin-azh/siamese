from .counter import Counter
from .timer import Timer


class IdentityTracker:
    MATCHED = 'matched'
    UN_MATCHED = 'un_matched'

    def __init__(self, name: str, max_time: float, max_conf: int):
        self._cnt = Counter()
        self._cnt.next()
        self._timer = Timer(max_time)
        self._id = name
        self._status = self.UN_MATCHED
        self._max_conf = max_conf

    @property
    def name(self):
        return self._id

    @property
    def status(self):
        return self._status

    def modify(self) -> None:
        self._timer.modify()
        if self.validated():
            self._status = self.MATCHED
        else:
            self._status = self.UN_MATCHED

    def __call__(self):
        self._cnt.next()
        if self._timer.validate():
            self.modify()
            return True
        else:
            return False

    def validated(self):
        return True if self._cnt() >= self._max_conf else False
