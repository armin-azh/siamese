import time


class Timer:
    """
    tracker timers
    """

    def __init__(self, threshold: float):
        self._time = time.time()
        self._threshold = threshold

    def modify(self) -> None:
        self._time = time.time()

    def diff(self) -> float:
        return time.time() - self._time

    def validate(self) -> bool:
        return True if self.diff() < self._threshold else False
