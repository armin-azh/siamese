# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog_recognition.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1000, 600)
        Dialog.setMinimumSize(QtCore.QSize(1000, 600))
        Dialog.setStyleSheet("background-color: rgb(35, 35, 35);")
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_left = QtWidgets.QFrame(self.frame)
        self.frame_left.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_left.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left.setObjectName("frame_left")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_left)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_recognition = QtWidgets.QLabel(self.frame_left)
        self.label_recognition.setMinimumSize(QtCore.QSize(640, 480))
        self.label_recognition.setMaximumSize(QtCore.QSize(640, 480))
        self.label_recognition.setStyleSheet("background-color: rgb(163, 255, 156);")
        self.label_recognition.setText("")
        self.label_recognition.setObjectName("label_recognition")
        self.verticalLayout.addWidget(self.label_recognition)
        self.frame_left_buttom = QtWidgets.QFrame(self.frame_left)
        self.frame_left_buttom.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_left_buttom.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_left_buttom.setObjectName("frame_left_buttom")
        self.verticalLayout.addWidget(self.frame_left_buttom)
        self.horizontalLayout_2.addWidget(self.frame_left)
        self.frame_right = QtWidgets.QFrame(self.frame)
        self.frame_right.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_right.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_right.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_right.setObjectName("frame_right")
        self.horizontalLayout_2.addWidget(self.frame_right)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
