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
        self.Btn_Toggle = QtWidgets.QPushButton(self.frame_toggle)
        self.Btn_Toggle.setStyleSheet("color: rgb(255, 255, 255);\n"
"border: 0px solid;")
        self.Btn_Toggle.setText("")
        self.Btn_Toggle.setObjectName("Btn_Toggle")
        self.verticalLayout_2.addWidget(self.Btn_Toggle)
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
        self.btn_identity = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_identity.setMinimumSize(QtCore.QSize(0, 50))
        self.btn_identity.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(85, 170, 255);\n"
"}")
        self.btn_identity.setObjectName("btn_identity")
        self.verticalLayout_4.addWidget(self.btn_identity)
        self.btn_setting = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_setting.setMinimumSize(QtCore.QSize(0, 50))
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
        self.btn_camera = QtWidgets.QPushButton(self.frame_top_menus)
        self.btn_camera.setMinimumSize(QtCore.QSize(0, 50))
        self.btn_camera.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"QPushButton:hover{\n"
"    background-color: rgb(85, 170, 255);\n"
"}")
        self.btn_camera.setObjectName("btn_camera")
        self.verticalLayout_4.addWidget(self.btn_camera)
        self.verticalLayout_3.addWidget(self.frame_top_menus, 0, QtCore.Qt.AlignTop)
        self.frame_buttom_menues = QtWidgets.QFrame(self.frame_left_menu)
        self.frame_buttom_menues.setMinimumSize(QtCore.QSize(0, 120))
        self.frame_buttom_menues.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_buttom_menues.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_buttom_menues.setObjectName("frame_buttom_menues")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_buttom_menues)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.btn_start = QtWidgets.QPushButton(self.frame_buttom_menues)
        self.btn_start.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.btn_start.setFont(font)
        self.btn_start.setStyleSheet("QPushButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(35, 35, 35);\n"
"    border: 0px solid;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"    background-color: rgb(229, 0, 0);\n"
"}")
        self.btn_start.setObjectName("btn_start")
        self.verticalLayout_6.addWidget(self.btn_start)
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
        self.image_label = QtWidgets.QLabel(self.camera_page)
        self.image_label.setGeometry(QtCore.QRect(150, 20, 640, 480))
        self.image_label.setMinimumSize(QtCore.QSize(640, 480))
        self.image_label.setStyleSheet("background-color: rgb(153, 255, 190);")
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")
        self.Pages.addWidget(self.camera_page)
        self.setting_page = QtWidgets.QWidget()
        self.setting_page.setObjectName("setting_page")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.setting_page)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.setting_left_frame = QtWidgets.QFrame(self.setting_page)
        self.setting_left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setting_left_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setting_left_frame.setObjectName("setting_left_frame")
        self.gridLayout = QtWidgets.QGridLayout(self.setting_left_frame)
        self.gridLayout.setObjectName("gridLayout")
        self.setting_table = QtWidgets.QTableWidget(self.setting_left_frame)
        self.setting_table.setMaximumSize(QtCore.QSize(500, 500))
        self.setting_table.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.setting_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setting_table.setGridStyle(QtCore.Qt.NoPen)
        self.setting_table.setObjectName("setting_table")
        self.setting_table.setColumnCount(3)
        self.setting_table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.setting_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.setting_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.setting_table.setHorizontalHeaderItem(2, item)
        self.gridLayout.addWidget(self.setting_table, 0, 0, 1, 1)
        self.horizontalLayout_4.addWidget(self.setting_left_frame)
        self.Pages.addWidget(self.setting_page)
        self.identity_page = QtWidgets.QWidget()
        self.identity_page.setObjectName("identity_page")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.identity_page)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.identity_form_frame = QtWidgets.QFrame(self.identity_page)
        self.identity_form_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.identity_form_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.identity_form_frame.setObjectName("identity_form_frame")
        self.horizontalLayout_3.addWidget(self.identity_form_frame)
        self.idnetity_camera_frame = QtWidgets.QFrame(self.identity_page)
        self.idnetity_camera_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.idnetity_camera_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.idnetity_camera_frame.setObjectName("idnetity_camera_frame")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.idnetity_camera_frame)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.iden_top_sub_frame = QtWidgets.QFrame(self.idnetity_camera_frame)
        self.iden_top_sub_frame.setMinimumSize(QtCore.QSize(60, 0))
        self.iden_top_sub_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.iden_top_sub_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.iden_top_sub_frame.setObjectName("iden_top_sub_frame")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.iden_top_sub_frame)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_record_camera = QtWidgets.QLabel(self.iden_top_sub_frame)
        self.label_record_camera.setMinimumSize(QtCore.QSize(100, 0))
        self.label_record_camera.setMaximumSize(QtCore.QSize(460, 300))
        self.label_record_camera.setStyleSheet("")
        self.label_record_camera.setText("")
        self.label_record_camera.setObjectName("label_record_camera")
        self.verticalLayout_9.addWidget(self.label_record_camera)
        self.verticalLayout_7.addWidget(self.iden_top_sub_frame)
        self.iden_buttom_sub_frame = QtWidgets.QFrame(self.idnetity_camera_frame)
        self.iden_buttom_sub_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.iden_buttom_sub_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.iden_buttom_sub_frame.setObjectName("iden_buttom_sub_frame")
        self.verticalLayout_7.addWidget(self.iden_buttom_sub_frame)
        self.horizontalLayout_3.addWidget(self.idnetity_camera_frame)
        self.Pages.addWidget(self.identity_page)
        self.start_page = QtWidgets.QWidget()
        self.start_page.setObjectName("start_page")
        self.image_label_start = QtWidgets.QLabel(self.start_page)
        self.image_label_start.setGeometry(QtCore.QRect(130, 20, 640, 480))
        self.image_label_start.setMinimumSize(QtCore.QSize(640, 480))
        self.image_label_start.setStyleSheet("background-color: rgb(35, 35, 35);\n"
"color: rgb(255, 255, 255);")
        self.image_label_start.setText("")
        self.image_label_start.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label_start.setObjectName("image_label_start")
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
        self.Pages.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition"))
        self.btn_identity.setText(_translate("MainWindow", "Identity"))
        self.btn_setting.setText(_translate("MainWindow", "Setting"))
        self.btn_camera.setText(_translate("MainWindow", "Camera"))
        self.btn_start.setText(_translate("MainWindow", "Start"))
        item = self.setting_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Header"))
        item = self.setting_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Name"))
        item = self.setting_table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Value"))
        self.actionIdentity.setText(_translate("MainWindow", "New Identity"))
        self.actionNew_Database.setText(_translate("MainWindow", "New Gallery"))
        self.actionProject_Settings.setText(_translate("MainWindow", "settings"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
