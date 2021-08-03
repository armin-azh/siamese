from datetime import datetime
from colorama import Fore
import pathlib


class Logger:
    def __init__(self, log_dir: pathlib.Path = None, *args, **kwargs):
        self._time_format = '[%H:%M:%S] '
        self._log_dir = log_dir.joinpath("summary.txt") if log_dir is not None else ""
        super(Logger, self).__init__(*args, **kwargs)

    def _now_time(self):
        return datetime.strftime(datetime.now(), self._time_format)

    def success(self,message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTGREEN_EX + time + Fore.GREEN + message + Fore.RESET
        print(status)

    def info(self, message, timestamp=True, white=False):
        time = ""
        if timestamp:
            time = self._now_time()

        if white:
            status = Fore.WHITE + time + Fore.WHITE + message + Fore.RESET

        else:
            status = Fore.LIGHTBLUE_EX + time + Fore.BLUE + message + Fore.RESET

        print(status)

    def warn(self, message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTYELLOW_EX + time + Fore.YELLOW + message + Fore.RESET
        print(status)

    def dang(self, message, timestamp=True):
        time = ""
        if timestamp:
            time = self._now_time()

        status = Fore.LIGHTRED_EX + time + Fore.RED + message + Fore.RESET
        print(status)

    def _write(self, data: dict, path: pathlib.Path) -> None:
        with open(str(path), 'a') as file:
            for key, value in data.items():
                file.write(f"{key} = {value}\n")

    def log(self, data: dict) -> None:
        self._write(data, self._log_dir)


class ExeLogger(Logger):
    def __init__(self, log_dir: pathlib.Path = None, *args, **kwargs):
        self._ex_log_dir = log_dir.joinpath("exe.log")
        super(ExeLogger, self).__init__(log_dir, *args, **kwargs)

    def exe_log(self, data: dict):
        self._write(data, self._ex_log_dir)
