class Counter:
    def __init__(self, limit=None):
        """
        if limit is none: the counter is infinite
        :param limit:
        """
        self._cnt = 0
        self._limit = limit

    def __is_infinite(self) -> bool:
        return True if self._limit is None else False

    def reset(self) -> None:
        self._cnt = 0

    def next(self) -> None:
        if self.__is_infinite() or self._cnt != self._limit-1:
            self._cnt += 1
        else:
            self._cnt = 0

    def __call__(self) -> int:
        return self._cnt
