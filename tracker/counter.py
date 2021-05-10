class Counter:
    def __init__(self):
        self._counter = 0

    def next(self) -> None:
        self._counter += 1

    def __call__(self) -> int:
        return self._counter

    def reset(self) -> None:
        self._counter = 0

