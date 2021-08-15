from pathlib import Path
from datetime import datetime

# loggers
from v2.tools.logger import get_logger, ConsoleLogger


class BasicService:
    def __init__(self, name, log_path: Path, *args, **kwargs):
        self._name = self.__class__.__name__ if name is None else name
        self._started_at = None
        self._log_path = log_path.joinpath(f"service/{self._name}")
        self._log_path.mkdir(parents=True, exist_ok=True)
        _cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self._log_filename = self._log_path.joinpath(f"{_cu}.log")
        self._file_logger = get_logger(self._log_filename, self._name)
        self._console_logger = ConsoleLogger()

        super(BasicService, self).__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    def exec_(self, *args, **kwargs) -> None:
        raise NotImplementedError
