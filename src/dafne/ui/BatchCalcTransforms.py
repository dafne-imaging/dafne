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
import numpy as np
from ..utils.RegistrationManager import RegistrationManager
from ..utils.dicomUtils.misc import dosma_volume_from_path
from ..utils.ThreadHelpers import separate_thread_decorator
from ..config import GlobalConfig
import sys


class CalcTransformWindow(QWidget, Ui_CalcTransformsUI):
    update_progress = pyqtSignal(int)
    success = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Dicom transform calculator")
        self.registrationManager = None
        self.update_progress.connect(self.set_progress)
        self.choose_Button.clicked.connect(self.load_data)
        self.calculate_button.clicked.connect(self.calculate)
        self.success.connect(self.show_success_box)
        self.data = None

    @pyqtSlot()
    def load_data(self):
        if GlobalConfig['ENABLE_NIFTI']:
            filter = 'Image files (*.dcm *.ima *.nii *.nii.gz);; Dicom files (*.dcm *.ima);;Nifti files (*.nii *.nii.gz);;All files (*.*)'
        else:
            filter = 'Dicom files (*.dcm *.ima);;All files (*.*)'

        dataFile, _ = QFileDialog.getOpenFileName(self, caption='Select dataset to import',
                                                  filter=filter)

        path = os.path.abspath(dataFile)
        print(path)
        _, ext = os.path.splitext(path)
        dataset_name = os.path.basename(path)

        containing_dir = os.path.dirname(path)
        
        if ext.lower() not in ['.nii', '.gz']:
            path = containing_dir
            
        
        medical_volume = None
        basename = ''
        try:
            medical_volume, affine_valid, title, basepath, basename = dosma_volume_from_path(path, self)
            self.data = medical_volume.volume
        except:
            pass
        print("Basepath", basepath)

        if self.data is None:
            self.progressBar.setValue(0)
            self.progressBar.setEnabled(False)
            self.calculate_button.setEnabled(False)
            QMessageBox.warning(self, 'Warning', 'Invalid dataset!')
            return

        self.data = medical_volume.volume
        data = list(np.transpose(self.data, [2, 0, 1]))

        self.progressBar.setMaximum(len(data))
        self.progressBar.setEnabled(True)
        self.registrationManager = RegistrationManager(data, None, os.getcwd(),
                                                       GlobalConfig['TEMP_DIR'])
        self.registrationManager.set_standard_transforms_name(basepath, basename)
        self.location_Text.setText(containing_dir if not basename else basename)
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
