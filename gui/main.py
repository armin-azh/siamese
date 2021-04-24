import sys
import os
import configparser
import pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from MainWindow import *

import cv2
import numpy as np
from settings import BASE_DIR, SETTINGS_HEADER


from thread import ClusterThread

base_path = pathlib.Path(BASE_DIR)


class MainWindow(QMainWindow):
    CAMERA_STATUS = {
        "start": True,
        "stop": False
    }
    CAMERA_START = 'start'
    CAMERA_STOP = 'stop'
    RECORD = False
    RECORD_FILENAME = './temp/xs1_tiger.avi'
    CAMERA = False

    def __init__(self):
        QMainWindow.__init__(self)

        # preprocessing
        if not pathlib.Path(self.RECORD_FILENAME).parent.exists():
            pathlib.Path(self.RECORD_FILENAME).parent.mkdir()

        # initialize main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/camera.png"))
        self.ui.btn_start_record.setIcon(QIcon("./icons/start.png"))
        self.ui.btn_stop_record.setIcon(QIcon("./icons/stop.png"))
        self.setFixedSize(self.size())

        # bind events
        self.ui.btn_camera.clicked.connect(self.on_press_camera_btn)
        self.ui.btn_identity.clicked.connect(self.on_press_identity_btn)
        self.ui.btn_setting.clicked.connect(self.on_press_setting_btn)
        self.ui.btn_start.clicked.connect(self.on_clicked_start_btn)
        self.ui.btn_start_record.clicked.connect(self.on_click_start_record_camera)
        self.ui.btn_stop_record.clicked.connect(self.on_click_stop_record_camera)

        # table
        self.ui.setting_table.setColumnWidth(0, 80)
        self.ui.setting_table.setColumnWidth(1, 100)
        self.ui.setting_table.setColumnWidth(2, 220)
        self.load_conf_to_table()

        # video
        self.camera = cv2.VideoCapture(0)
        self.camera_timer = QtCore.QTimer()
        self.frame_streamer = VideoSteamer()
        self.frame_recorder = None

        # thread
        self.clustering = ClusterThread()

        self.show()

    # slots
    def frame_updater_slot(self, image):
        self.ui.image_label_start.setPixmap(QtGui.QPixmap.fromImage(image))

    def frame_update_record_slot(self, image):
        image = image.scaled(460, 300, Qt.KeepAspectRatio)
        self.ui.label_record_camera.setPixmap(QtGui.QPixmap.fromImage(image))

    def frame_np_array_save_slot(self, frame):
        """
        slot for save frame in video writer
        :param frame:
        :return:
        """
        self.write_frame(frame)

    def cluster_slot(self):
        pass

    # events
    def on_click_start_record_camera(self):
        """
        start recording and open camera
        :return:
        """
        if not self.RECORD:
            if not self.CAMERA:
                self.frame_streamer.start()
                self.CAMERA = True
            self.frame_recorder = self.initiate_video_writer()
            self.frame_streamer.frame_update.connect(self.frame_np_array_save_slot)
            self.RECORD = True

    def on_click_stop_record_camera(self):
        """
        stop recording and close camera
        :return:
        """
        if self.RECORD and self.CAMERA:
            self.frame_streamer.stop()
            self.frame_recorder.release()
            self.RECORD = False
            self.CAMERA = False
            self.clustering.start()
            self.clustering.cluster_signal.connect(self.cluster_slot)

    def on_press_camera_btn(self):
        self.ui.Pages.setCurrentIndex(0)

    def on_press_identity_btn(self):
        self.ui.Pages.setCurrentIndex(2)
        self.setWindowTitle("Identity")
        self.frame_streamer.start()
        self.frame_streamer.image_update.connect(self.frame_update_record_slot)

    def on_press_setting_btn(self):
        self.ui.Pages.setCurrentIndex(1)

    def on_clicked_start_btn(self):
        self.setWindowTitle("Recognizing")
        self.ui.Pages.setCurrentIndex(3)
        if not self.CAMERA:
            self.ui.btn_setting.setEnabled(False)
            self.ui.btn_camera.setEnabled(False)
            self.ui.btn_identity.setEnabled(False)
            self.ui.btn_start.setText(self.CAMERA_STOP)
            self.frame_streamer.start()
            self.frame_streamer.image_update.connect(self.frame_updater_slot)
            self.CAMERA = True
        else:
            self.ui.btn_start.setText(self.CAMERA_START)
            self.frame_streamer.stop()
            self.CAMERA = False
            self.ui.btn_setting.setEnabled(True)
            self.ui.btn_camera.setEnabled(True)
            self.ui.btn_identity.setEnabled(True)
            self.ui.btn_start.setText(self.CAMERA_START)

    # methods
    def initiate_video_writer(self):
        return cv2.VideoWriter(self.RECORD_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))

    def write_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frame_recorder.write(frame)

    def load_conf_to_table(self):
        """
        load configuration file to the table
        :return:
        """
        conf = configparser.ConfigParser()
        conf.read(base_path.joinpath('conf.ini'))

        data = []
        for header in SETTINGS_HEADER:
            h_con = conf[header]
            for key, value in h_con.items():
                data.append((header, key, value))

        self.ui.setting_table.setRowCount(len(data))
        row = 0
        for h, k, v in data:
            self.ui.setting_table.setItem(row, 0, QtWidgets.QTableWidgetItem(h))
            self.ui.setting_table.setItem(row, 1, QtWidgets.QTableWidgetItem(k))
            self.ui.setting_table.setItem(row, 2, QtWidgets.QTableWidgetItem(v))
            row += 1


class VideoSteamer(QtCore.QThread):
    image_update = QtCore.pyqtSignal(QtGui.QImage)
    frame_update = QtCore.pyqtSignal(np.ndarray)
    thread_active = None

    def run(self):
        self.thread_active = True
        cap = cv2.VideoCapture(0)
        while self.thread_active:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                qt_frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.image_update.emit(qt_frame)
                self.frame_update.emit(frame)
        cap.release()

    def stop(self):
        self.thread_active = False
        self.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
