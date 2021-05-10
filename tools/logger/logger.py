from datetime import datetime
from colorama import Fore


class Logger:
    def __init__(self, log_dir=None):
        self._time_format = '[%H:%M:%S] '
        self._log_dir = log_dir

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
