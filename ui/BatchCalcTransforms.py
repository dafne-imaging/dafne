#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import QWidget, QMainWindow, QFileDialog, QMessageBox, QApplication
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from .CalcTransformsUI import Ui_CalcTransformsUI
import os
import sys
import numpy as np
from utils.RegistrationManager import RegistrationManager
from utils.dicomUtils.dicom3D import load3dDicom
from utils.ThreadHelpers import separate_thread_decorator
from config import GlobalConfig
import sys

TRANSFORM_FILENAME = 'transforms.p'

class CalcTransformWindow(QWidget, Ui_CalcTransformsUI):

    update_progress = pyqtSignal(int)
    success = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Dicom transform calculator")
        self.transform_filename = None
        self.registrationManager = None
        self.update_progress.connect(self.set_progress)
        self.choose_Button.clicked.connect(self.load_data)
        self.calculate_button.clicked.connect(self.calculate)
        self.success.connect(self.show_success_box)

    @pyqtSlot()
    def load_data(self):
        filter = 'Dicom files (*.dcm *.ima);;All files (*.*)'

        dataFile, _ = QFileDialog.getOpenFileName(self, caption='Select dataset to import',
                                                  filter=filter)

        path = os.path.abspath(dataFile)
        containing_dir = os.path.dirname(path)

        self.data, _ = load3dDicom(containing_dir)
        if self.data is None:
            self.progressBar.setValue(0)
            self.progressBar.setEnabled(False)
            self.calculate_button.setEnabled(False)
            QMessageBox.warning(self, 'Warning', 'Invalid DICOM dataset! Select a dicom file')
            return

        data = list(np.transpose(self.data, [2, 0, 1]))
        self.progressBar.setMaximum(len(data))
        self.progressBar.setEnabled(True)
        self.transform_filename = os.path.join(containing_dir, TRANSFORM_FILENAME)
        self.registrationManager = RegistrationManager(data, self.transform_filename, os.getcwd(), GlobalConfig['TEMP_DIR'])
        self.location_Text.setText(containing_dir)
        self.calculate_button.setEnabled(True)

    @pyqtSlot(int)
    def set_progress(self, value):
        self.progressBar.setValue(value)

    @pyqtSlot()
    def show_success_box(self):
        QMessageBox.information(self, 'Done', 'Done!')

    @pyqtSlot()
    @separate_thread_decorator
    def calculate(self):
        self.choose_Button.setEnabled(False)
        self.calculate_button.setEnabled(False)
        self.registrationManager.calc_transforms(lambda value: self.update_progress.emit(value))
        self.choose_Button.setEnabled(True)
        self.calculate_button.setEnabled(False)
        self.update_progress.emit(0)
        self.success.emit()


def run():
    app = QApplication(sys.argv)
    window = QMainWindow()
    widget = CalcTransformWindow()
    window.setCentralWidget(widget)
    window.setWindowTitle("Dicom transform calculator")
    window.show()
    sys.exit(app.exec_())
