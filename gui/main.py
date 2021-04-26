import sys
import pathlib
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import qtawesome as qta
from MainWindow import *
from dialog import Ui_Dialog

import cv2
from settings import BASE_DIR, DEFAULT_CONF, GALLERY_CONF

from thread import VideoSteamerThread
from style import BTN_CAM_DEF, BTN_CAM_LOCK
from database.component import ImageDatabase

base_path = pathlib.Path(BASE_DIR)


class MainWindow(QMainWindow):
    RECORD_ON = False
    CAMERA_ON = False
    RECORD_FILENAME = pathlib.Path(BASE_DIR).joinpath(DEFAULT_CONF.get("save_video"))
    DIALOG = None

    def __init__(self):
        QMainWindow.__init__(self)

        # database
        self.database = ImageDatabase(db_path=str(base_path.joinpath(GALLERY_CONF.get("database_path"))))

        # preprocessing
        if not self.RECORD_FILENAME.exists():
            self.RECORD_FILENAME.mkdir()

        # initialize main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/camera.png"))
        self.ui.Pages.setCurrentIndex(0)

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
            self.thread_video_stream.record_status.connect(self.slot_video_record_status)

    def on_click_record_start(self):
        self.ui.Pages.setCurrentIndex(0)
        if not self.RECORD_ON:
            dialog = QtWidgets.QDialog()
            ui = Ui_Dialog()
            ui.setupUi(dialog)
            self.DIALOG = ui
            dialog.setWindowTitle("Add New Identity")
            dialog.setWindowIcon(qta.icon('fa5.user'))
            dialog.show()
            ui.line_identity_name.textChanged.connect(self.on_change_text)
            res = dialog.exec_()
            identity_name = ui.line_identity_name.text()
            identity_name = identity_name.title()
            if res == QtWidgets.QDialog.Accepted and identity_name:
                if not self.CAMERA_ON:
                    self.video_streamer_start()
                self.video_writer_initializer(identity_name)
                # self.thread_video_stream.frame_update.connect(self.slot_video_writer_frame)
                self.change_btn_start_record_status()

    def on_click_record_stop(self):
        self.video_writer_release()

    def on_change_text(self):
        if self.DIALOG is not None:
            if self.check_identity_name(self.DIALOG.line_identity_name.text().title()):
                self.DIALOG.label_identity_name_status.setText("Exists")
            else:
                self.DIALOG.label_identity_name_status.setText("Not Exists")

    # slots
    # def slot_video_writer_frame(self, frame):
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     self.video_writer.write(frame)

    def slot_video_frame_qt(self, qt_frame):
        self.ui.label_camera.setPixmap(QtGui.QPixmap(qt_frame))

    def slot_video_record_status(self, status):
        if not status:
            self.video_writer_release()

    # methods
    def video_streamer_start(self):
        self.thread_video_stream.start()
        self.CAMERA_ON = True

    def video_streamer_shutdown(self):
        if self.CAMERA_ON:
            self.thread_video_stream.stop()
            self.CAMERA_ON = False

    def video_writer_initializer(self, identity_name):
        filename = self.RECORD_FILENAME.joinpath(identity_name + '.avi')
        self.video_writer = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))
        self.thread_video_stream.record_cap = self.video_writer
        self.RECORD_ON = True
        self.thread_video_stream.record_on = True

    def video_writer_release(self):
        if self.video_writer is not None:
            self.RECORD_ON = False
            self.thread_video_stream.record_on = False
            self.video_writer.release()
            self.change_btn_stop_record_status()

    def change_btn_start_record_status(self):
        self.ui.btn_cam.setEnabled(False)
        self.ui.btn_setting.setEnabled(False)
        self.ui.btn_gallery.setEnabled(False)

    def change_btn_stop_record_status(self):
        self.ui.btn_cam.setEnabled(True)
        self.ui.btn_setting.setEnabled(True)
        self.ui.btn_gallery.setEnabled(True)

    def check_identity_name(self, identity_name):
        ids = list(self.database.get_identity_image_paths().keys())
        return True if identity_name in ids else False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
