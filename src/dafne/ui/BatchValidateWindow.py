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
import os
import sys

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QApplication, QMainWindow, QInputDialog

from . import GenericInputDialog
from .ValidateUI import Ui_ValidateUI
from ..utils.ThreadHelpers import separate_thread_decorator
from ..config import config

from ..utils.BatchValidator import BatchValidator


class BatchValidateWindow(QWidget, Ui_ValidateUI, BatchValidator):
    update_progress = pyqtSignal(int, int, str)
    update_overall_progress = pyqtSignal(int, int)
    alert_signal = pyqtSignal(str)
    success = pyqtSignal()

    def __init__(self, **kwargs):
        QWidget.__init__(self)
        BatchValidator.__init__(self, **kwargs)
        self.setupUi(self)
        self.setWindowTitle("Batch model validation")
        self.update_progress.connect(self.set_progress)
        self.update_overall_progress.connect(self.set_overall_progress)
        self.alert_signal.connect(self.alert)
        self.data_choose_Button.clicked.connect(self.choose_data)
        self.mask_choose_Button.clicked.connect(self.choose_mask_dir)
        self.roi_choose_Button.clicked.connect(self.do_load_roi)
        self.evaluate_Button.clicked.connect(self.start_calculation)
        self.configure_button.clicked.connect(self.configure)


    def configure(self):
        accepted, values = GenericInputDialog.show_dialog('Configure', [
            GenericInputDialog.TextLineInput('Classification', self.classification),
            GenericInputDialog.TextLineInput('Timestamp start', str(self.timestamp_interval[0])),
            GenericInputDialog.TextLineInput('Timestamp end', str(self.timestamp_interval[1])),
            GenericInputDialog.BooleanInput('Upload stats', self.upload_stats),
            GenericInputDialog.BooleanInput('Save local', self.save_local),
            GenericInputDialog.TextLineInput('Local filename', self.local_filename),
        ])

        if accepted:
            self.classification = values[0]
            self.timestamp_interval = (int(values[1]), int(values[2]))
            self.upload_stats = values[3]
            self.save_local = values[4]
            self.local_filename = values[5]

    def loadDirectory_internal(self, path):
        basename = BatchValidator.loadDirectory_internal(self, path)
        self.data_location_Text.setText(basename)

    def load_directory(self, path):
        self.data_choose_Button.setEnabled(False)
        BatchValidator.load_directory(self, path)
        self.data_location_Text.setText(self.basepath)
        self.mask_choose_Button.setEnabled(True)
        self.roi_choose_Button.setEnabled(True)

        self.data_choose_Button.setEnabled(True)

        if self.mask_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)

        if self.mask_list:
            self.mask_location_Text.setText(self.basename)
            self.mask_choose_Button.setEnabled(False)
            self.roi_choose_Button.setEnabled(False)

    def signal_progress(self, value, maximum, msg=''):
        self.update_progress.emit(value, maximum, msg)

    @pyqtSlot(int, int, str)
    def set_progress(self, value, maximum, msg):
        self.slice_progressBar.setMaximum(maximum)
        self.slice_progressBar.setValue(value)
        self.status_Label.setText(msg)

    def signal_overall_progress(self, value, maximum):
        self.update_overall_progress.emit(value, maximum)

    @pyqtSlot(int, int)
    def set_overall_progress(self, value, maximum):
        self.overall_progressBar.setMaximum(maximum)
        self.overall_progressBar.setValue(value)

    def signal_alert(self, msg):
        self.alert_signal.emit(msg)

    @pyqtSlot(str)
    def alert(self, text):
        QMessageBox.warning(self, "Warning", text, QMessageBox.Ok)

    @pyqtSlot()
    def choose_data(self):
        if config.GlobalConfig['ENABLE_NIFTI']:
            filter = 'Image files (*.dcm *.ima *.nii *.nii.gz *.npy *.npz);;Dicom files (*.dcm *.ima);;Nifti files (*.nii *.nii.gz);;Numpy files (*.npy);;Data + Mask bundle (*npz);;All files ()'
        else:
            filter = 'Image files (*.dcm *.ima *.npy *.npz);;Dicom files (*.dcm *.ima);;Numpy files (*.npy);;Data + Mask bundle (*npz);;All files ()'

        dataFile, _ = QFileDialog.getOpenFileName(self, caption='Select dataset to import',
                                                  filter=filter)
        if dataFile:
            self.im_list = []
            self.mask_list = {}
            self.mask_choose_Button.setEnabled(False)
            self.roi_choose_Button.setEnabled(False)
            self.data_location_Text.setText("")
            self.mask_location_Text.setText("")
            self.load_directory(dataFile)


    @pyqtSlot()
    def do_load_roi(self):
        roiPickleName, _ = QFileDialog.getOpenFileName(self, caption='Select ROI file',
                                                       filter="Pickle files (*.p)")

        if roiPickleName is None:
            return

        self.mask_location_Text.setText(roiPickleName)
        self.loadROIPickle(roiPickleName)

    @separate_thread_decorator
    def loadROIPickle(self, roiPickleName):
        self.mask_choose_Button.setEnabled(False)
        self.roi_choose_Button.setEnabled(False)
        self.data_choose_Button.setEnabled(False)
        if not BatchValidator.loadROIPickle(self, roiPickleName):
            return

        self.evaluate_Button.setEnabled(True)
        self.mask_choose_Button.setEnabled(True)
        self.roi_choose_Button.setEnabled(True)
        self.data_choose_Button.setEnabled(True)
        self.signal_progress(0, 2)


    @pyqtSlot()
    def choose_mask_dir(self):
        maskDir = QFileDialog.getExistingDirectory(self, caption='Select folder containing other DICOM folders or Nifti files')

        if maskDir:
            self.mask_import(maskDir)

    @separate_thread_decorator
    def mask_import(self, filename):
        self.mask_choose_Button.setEnabled(False)
        self.roi_choose_Button.setEnabled(False)
        self.data_choose_Button.setEnabled(False)
        BatchValidator.mask_import(self, filename)
        if self.data_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)

        self.mask_choose_Button.setEnabled(True)
        self.roi_choose_Button.setEnabled(True)
        self.data_choose_Button.setEnabled(True)


    @pyqtSlot()
    def start_calculation(self):
        n_slices = len(self.mask_list)
        n_rois = len(next(iter(self.mask_list.values()))) # get the number of rois from an arbitrary slice
        accept = QMessageBox.question(self, "Run validation?", f'Running validation for model {self.classification} on {len(self.timestamps_to_download)} models\nwith {n_rois} ROIs over {n_slices} slices. Continue?', QMessageBox.Yes | QMessageBox.No)
        if accept == QMessageBox.No:
            return
        if not self.batch_mode:
            comment, ok = QInputDialog.getText(self, "Add comment", "Add an identifier for this dataset")
            if not ok:
                return
        else:
            comment = ''

        self.setEnabled(False)
        self.calculate(comment)

    @separate_thread_decorator
    def calculate(self, comment):
        BatchValidator.calculate(self, comment)
        self.signal_overall_progress(0, 1)
        self.setEnabled(True)

    def mask_import(self, filename):
        path = os.path.abspath(filename)
        _, ext = os.path.splitext(path)
        self.mask_location_Text.setText(path)
        BatchValidator.mask_import(self, filename)


def run(path = None, roi = None, **kwargs):
    app = QApplication(sys.argv)
    window = QMainWindow()
    widget = BatchValidateWindow()
    window.setCentralWidget(widget)
    window.setWindowTitle("Batch validation")
    window.show()
    if path is not None:
        widget.batch_mode = True
        widget.load_directory(path)
    if roi is not None:
        widget.loadROIPickle(roi)

    sys.exit(app.exec_())
