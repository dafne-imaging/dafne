# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LogWindowUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LogWindow(object):
    def setupUi(self, LogWindow):
        LogWindow.setObjectName("LogWindow")
        LogWindow.resize(885, 519)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(LogWindow)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(LogWindow)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.output_text = QtWidgets.QPlainTextEdit(LogWindow)
        self.output_text.setUndoRedoEnabled(False)
        self.output_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.output_text.setReadOnly(True)
        self.output_text.setObjectName("output_text")
        self.verticalLayout.addWidget(self.output_text)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(LogWindow)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.error_text = QtWidgets.QPlainTextEdit(LogWindow)
        self.error_text.setUndoRedoEnabled(False)
        self.error_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.error_text.setReadOnly(True)
        self.error_text.setObjectName("error_text")
        self.verticalLayout_2.addWidget(self.error_text)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.refresh_btn = QtWidgets.QPushButton(LogWindow)
        self.refresh_btn.setObjectName("refresh_btn")
        self.verticalLayout_3.addWidget(self.refresh_btn)

        self.retranslateUi(LogWindow)
        QtCore.QMetaObject.connectSlotsByName(LogWindow)

    def retranslateUi(self, LogWindow):
        _translate = QtCore.QCoreApplication.translate
        LogWindow.setWindowTitle(_translate("LogWindow", "Form"))
        self.label.setText(_translate("LogWindow", "Output:"))
        self.label_2.setText(_translate("LogWindow", "Error:"))
        self.refresh_btn.setText(_translate("LogWindow", "Refresh"))
