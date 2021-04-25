import sys
import os
import configparser
import pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import qtawesome as qta
from MainWindow import *

import cv2
from settings import BASE_DIR, SETTINGS_HEADER

from thread import ClusterThread, VideoSteamerThread

base_path = pathlib.Path(BASE_DIR)


class MainWindow(QMainWindow):
    RECORD_ON = False
    CAMERA_ON = False
    RECORD_FILENAME = './temp/xs1_tiger.avi'

    def __init__(self):
        QMainWindow.__init__(self)

        # database

        # preprocessing
        if not pathlib.Path(self.RECORD_FILENAME).parent.exists():
            pathlib.Path(self.RECORD_FILENAME).parent.mkdir()

        # initialize main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/camera.png"))

        # btn
        self.ui.btn_start_record.setIcon(QIcon("./icons/start.png"))
        self.ui.btn_stop_record.setIcon(QIcon("./icons/stop.png"))

        # btn event bind
        self.ui.btn_cam.clicked.connect(self.on_click_camera)
        self.ui.btn_start_record.clicked.connect(self.on_click_record_start)
        self.ui.btn_stop_record.clicked.connect(self.on_click_record_stop)

        # thread
        self.thread_video_stream = VideoSteamerThread()

        # video writer
        self.video_writer = None

        self.show()

    # events
    def on_click_camera(self):
        self.ui.Pages.setCurrentIndex(0)
        if not self.CAMERA_ON:
            self.video_streamer_start()
            self.thread_video_stream.image_update.connect(self.slot_video_frame_qt)

    def on_click_record_start(self):
        self.ui.Pages.setCurrentIndex(0)
        if not self.RECORD_ON:
            if not self.CAMERA_ON:
                self.video_streamer_start()
            self.video_writer_initializer()
            self.thread_video_stream.frame_update.connect(self.slot_video_writer_frame)

    def on_click_record_stop(self):
        self.video_writer_release()

    # slots
    def slot_video_writer_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def slot_video_frame_qt(self, qt_frame):
        self.ui.label_camera.setPixmap(QtGui.QPixmap(qt_frame))

    # methods
    def video_streamer_start(self):
        self.thread_video_stream.start()
        self.CAMERA_ON = True

    def video_streamer_shutdown(self):
        if self.CAMERA_ON:
            self.thread_video_stream.stop()
            self.CAMERA_ON = False

    def video_writer_initializer(self):
        self.video_writer = cv2.VideoWriter(self.RECORD_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))
        self.RECORD_ON = True
        self.thread_video_stream.record_on = True

    def video_writer_release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.RECORD_ON = False
            self.thread_video_stream.record_on = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
