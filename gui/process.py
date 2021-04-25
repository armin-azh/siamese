import pathlib

from PyQt5.QtCore import QProcess

from settings import BASE_DIR


class RecognitionProcess(QProcess):
    def __init__(self):
        super(RecognitionProcess, self).__init__()

    print(pathlib.Path(BASE_DIR).joinpath("manage.py"))

    def start_realtime_recognition(self):
        self.start("python", [pathlib.Path(BASE_DIR).joinpath("manage.py"), "--realtime"])
        print(bytes(self.readAllStandardError()))
        print(bytes(self.readAllStandardOutput()))