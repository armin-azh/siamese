from datetime import datetime
from colorama import Fore
import pathlib


class Logger:
    def __init__(self, log_dir: pathlib.Path = None):
        self._time_format = '[%H:%M:%S] '
        self._log_dir = log_dir.joinpath("summary.txt") if log_dir is not None else ""

    def _now_time(self):
        return datetime.strftime(datetime.now(), self._time_format)

    def info(self, message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTBLUE_EX + time + Fore.BLUE + message
        print(status)

    def warn(self, message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTYELLOW_EX + time + Fore.YELLOW + message
        print(status)

    def dang(self, message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTRED_EX + time + Fore.RED + message
        print(status)

    def _write(self, data: dict) -> None:
        with open(str(self._log_dir), 'a') as file:
            for key, value in data.items():
                file.write(f"{key} = {value}\n")

    def log(self, data: dict) -> None:
        self._write(data)
