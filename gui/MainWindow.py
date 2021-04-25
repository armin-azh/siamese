# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1000, 600)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 600))
        MainWindow.setStyleSheet("background-color: rgb(45, 45, 45);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Top_Bar = QtWidgets.QFrame(self.centralwidget)
        self.Top_Bar.setMaximumSize(QtCore.QSize(16777215, 40))
        self.Top_Bar.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.Top_Bar.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.Top_Bar.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Top_Bar.setObjectName("Top_Bar")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.Top_Bar)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_toggle = QtWidgets.QFrame(self.Top_Bar)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_toggle.sizePolicy().hasHeightForWidth())
        self.frame_toggle.setSizePolicy(sizePolicy)
        self.frame_toggle.setMaximumSize(QtCore.QSize(70, 40))
        self.frame_toggle.setStyleSheet("background-color: rgb(77, 154, 231);")
        self.frame_toggle.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_toggle.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_toggle.setObjectName("frame_toggle")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_toggle)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_toggle = QtWidgets.QPushButton(self.frame_toggle)
        self.btn_toggle.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 0px solid;")
        self.btn_toggle.setText("")
        self.btn_toggle.setObjectName("btn_toggle")
        self.verticalLayout_2.addWidget(self.btn_toggle)
        self.horizontalLayout.addWidget(self.frame_toggle)
        self.frame_top = QtWidgets.QFrame(self.Top_Bar)
        self.frame_top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout.addWidget(self.frame_top)
        self.verticalLayout.addWidget(self.Top_Bar)
        self.Content = QtWidgets.QFrame(self.centralwidget)
        self.Content.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.Content.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Content.setObjectName("Content")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.Content)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_left_menu = QtWidgets.QFrame(self.Content)
        self.frame_left_menu.setMinimumSize(QtCore.QSize(70, 0))
        self.frame_left_menu.setMaximumSize(QtCore.QSize(70, 16777215))
        self.frame_left_menu.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.frame_left_menu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left_menu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left_menu.setObjectName("frame_left_menu")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_left_menu)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_top_menus = QtWidgets.QFrame(self.frame_left_menu)
        self.frame_top_menus.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top_menus.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_menus.setObjectName("frame_top_menus")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_top_menus)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btn_cam = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_cam.setMinimumSize(QtCore.QSize(0, 60))
        self.btn_cam.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(85, 170, 255);\n"
"}")
        self.btn_cam.setObjectName("btn_cam")
        self.verticalLayout_4.addWidget(self.btn_cam)
        self.btn_gallery = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_gallery.setMinimumSize(QtCore.QSize(0, 60))
        self.btn_gallery.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(85, 170, 255);\n"
"}")
        self.btn_gallery.setObjectName("btn_gallery")
        self.verticalLayout_4.addWidget(self.btn_gallery)
        self.btn_setting = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_setting.setMinimumSize(QtCore.QSize(0, 60))
        self.btn_setting.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(85, 170, 255);\n"
"}")
        self.btn_setting.setObjectName("btn_setting")
        self.verticalLayout_4.addWidget(self.btn_setting)
        self.verticalLayout_3.addWidget(self.frame_top_menus, 0, QtCore.Qt.AlignTop)
        self.frame_buttom_menues = QtWidgets.QFrame(self.frame_left_menu)
        self.frame_buttom_menues.setMinimumSize(QtCore.QSize(0, 80))
        self.frame_buttom_menues.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_buttom_menues.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_buttom_menues.setObjectName("frame_buttom_menues")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_buttom_menues)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.btn_start_record = QtWidgets.QPushButton(self.frame_buttom_menues)
        self.btn_start_record.setMinimumSize(QtCore.QSize(0, 40))
        self.btn_start_record.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background-color: rgb(45, 45, 45);\n"
"}")
        self.btn_start_record.setText("")
        self.btn_start_record.setObjectName("btn_start_record")
        self.verticalLayout_6.addWidget(self.btn_start_record)
        self.btn_stop_record = QtWidgets.QPushButton(self.frame_buttom_menues)
        self.btn_stop_record.setMinimumSize(QtCore.QSize(0, 40))
        self.btn_stop_record.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background-color: rgb(45, 45, 45);\n"
"}")
        self.btn_stop_record.setText("")
        self.btn_stop_record.setObjectName("btn_stop_record")
        self.verticalLayout_6.addWidget(self.btn_stop_record)
        self.verticalLayout_3.addWidget(self.frame_buttom_menues, 0, QtCore.Qt.AlignBottom)
        self.horizontalLayout_2.addWidget(self.frame_left_menu)
        self.frame_pages = QtWidgets.QFrame(self.Content)
        self.frame_pages.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_pages.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_pages.setObjectName("frame_pages")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_pages)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.Pages = QtWidgets.QStackedWidget(self.frame_pages)
        self.Pages.setObjectName("Pages")
        self.camera_page = QtWidgets.QWidget()
        self.camera_page.setObjectName("camera_page")
        self.label_camera = QtWidgets.QLabel(self.camera_page)
        self.label_camera.setGeometry(QtCore.QRect(140, 30, 640, 480))
        self.label_camera.setMinimumSize(QtCore.QSize(640, 480))
        self.label_camera.setText("")
        self.label_camera.setObjectName("label_camera")
        self.Pages.addWidget(self.camera_page)
        self.setting_page = QtWidgets.QWidget()
        self.setting_page.setObjectName("setting_page")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.setting_page)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.Pages.addWidget(self.setting_page)
        self.identity_page = QtWidgets.QWidget()
        self.identity_page.setObjectName("identity_page")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.identity_page)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.Pages.addWidget(self.identity_page)
        self.start_page = QtWidgets.QWidget()
        self.start_page.setObjectName("start_page")
        self.Pages.addWidget(self.start_page)
        self.verticalLayout_5.addWidget(self.Pages)
        self.horizontalLayout_2.addWidget(self.frame_pages)
        self.verticalLayout.addWidget(self.Content)
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionNew_Database = QtWidgets.QAction(MainWindow)
        self.actionNew_Database.setObjectName("actionNew_Database")
        self.actionProject_Settings = QtWidgets.QAction(MainWindow)
        self.actionProject_Settings.setObjectName("actionProject_Settings")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")

        self.retranslateUi(MainWindow)
        self.Pages.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition"))
        self.btn_cam.setText(_translate("MainWindow", "Camera"))
        self.btn_gallery.setText(_translate("MainWindow", "Gallery"))
        self.btn_setting.setText(_translate("MainWindow", "Setting"))
        self.actionIdentity.setText(_translate("MainWindow", "New Identity"))
        self.actionNew_Database.setText(_translate("MainWindow", "New Gallery"))
        self.actionProject_Settings.setText(_translate("MainWindow", "settings"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
