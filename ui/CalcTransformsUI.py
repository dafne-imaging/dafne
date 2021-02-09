# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CalcTransformsUI.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CalcTransformsUI(object):
    def setupUi(self, CalcTransformsUI):
        CalcTransformsUI.setObjectName("CalcTransformsUI")
        CalcTransformsUI.resize(400, 176)
        self.verticalLayout = QtWidgets.QVBoxLayout(CalcTransformsUI)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(CalcTransformsUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.location_Text = QtWidgets.QLineEdit(CalcTransformsUI)
        self.location_Text.setEnabled(False)
        self.location_Text.setObjectName("location_Text")
        self.horizontalLayout.addWidget(self.location_Text)
        self.choose_Button = QtWidgets.QPushButton(CalcTransformsUI)
        self.choose_Button.setObjectName("choose_Button")
        self.horizontalLayout.addWidget(self.choose_Button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(CalcTransformsUI)
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        spacerItem = QtWidgets.QSpacerItem(20, 45, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.calculate_button = QtWidgets.QPushButton(CalcTransformsUI)
        self.calculate_button.setEnabled(False)
        self.calculate_button.setObjectName("calculate_button")
        self.verticalLayout.addWidget(self.calculate_button)

        self.retranslateUi(CalcTransformsUI)
        QtCore.QMetaObject.connectSlotsByName(CalcTransformsUI)

    def retranslateUi(self, CalcTransformsUI):
        _translate = QtCore.QCoreApplication.translate
        CalcTransformsUI.setWindowTitle(_translate("CalcTransformsUI", "Form"))
        self.label.setText(_translate("CalcTransformsUI", "Location:"))
        self.choose_Button.setText(_translate("CalcTransformsUI", "Choose..."))
        self.calculate_button.setText(_translate("CalcTransformsUI", "Calculate Transforms"))
