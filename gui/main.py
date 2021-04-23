import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from MainWindow import *

import cv2


class MainWindow(QMainWindow):
    CAMERA_STATUS = {
        "start":True,
        "stop":False
    }
    CAMERA_START = 'start'
    CAMERA_STOP = 'stop'

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/camera.png"))
        self.ui.btn_start_record.setIcon(QIcon("./icons/start.png"))
        self.ui.btn_stop_record.setIcon(QIcon("./icons/stop.png"))
        self.setFixedSize(self.size())

        self.ui.btn_camera.clicked.connect(self.on_press_camera_btn)
        self.ui.btn_identity.clicked.connect(self.on_press_identity_btn)
        self.ui.btn_setting.clicked.connect(self.on_press_setting_btn)
        self.ui.btn_start.clicked.connect(self.on_clicked_start_btn)

        self.camera = cv2.VideoCapture(0)
        self.camera_timer = QtCore.QTimer()

        self.frame_streamer = VideoSteamer()

        self.show()

    def frame_updater_slot(self, image):
        self.ui.image_label_start.setPixmap(QtGui.QPixmap.fromImage(image))

    def frame_update_record_slot(self,image):
        image = image.scaled(460,300,Qt.KeepAspectRatio)
        self.ui.label_record_camera.setPixmap(QtGui.QPixmap.fromImage(image))

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
        if self.CAMERA_STATUS.get(self.ui.btn_start.text().lower()):
            self.ui.btn_setting.setEnabled(False)
            self.ui.btn_camera.setEnabled(False)
            self.ui.btn_identity.setEnabled(False)
            self.ui.btn_start.setText(self.CAMERA_STOP)
            self.frame_streamer.start()
            self.frame_streamer.image_update.connect(self.frame_updater_slot)
        else:
            self.ui.btn_start.setText(self.CAMERA_START)
            self.frame_streamer.stop()
            self.ui.btn_setting.setEnabled(True)
            self.ui.btn_camera.setEnabled(True)
            self.ui.btn_identity.setEnabled(True)
            self.ui.btn_start.setText(self.CAMERA_START)


class VideoSteamer(QtCore.QThread):
    image_update = QtCore.pyqtSignal(QtGui.QImage)
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
        cap.release()

    def stop(self):
        self.thread_active = False
        self.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
