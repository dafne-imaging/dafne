#  Copyright (c) 2021 Dafne-Imaging Team
import os
import sys

import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QApplication, QMainWindow

from config import config
from ui import GenericInputDialog
from ui.ValidateUI import Ui_ValidateUI
from ui.pyDicomView import ImListProxy
from utils.ThreadHelpers import separate_thread_decorator
from utils.dicomUtils.misc import dosma_volume_from_path


class BatchValidateWindow(QWidget, Ui_ValidateUI):
    update_progress = pyqtSignal(int, int)
    alert_signal = pyqtSignal(str)
    success = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Batch model validation")
        self.update_progress.connect(self.set_progress)
        self.alert_signal.connect(self.alert)
        self.data_choose_Button.clicked.connect(self.choose_data)
        self.resolution = [1,1,1]
        self.resolution_valid = False
        self.medical_volume = None
        self.imList = []
        self.basepath = ''
        self.basename = ''
        self.maskpath = ''
        self.mask_list = {}


        self.evaluate_Button.clicked.connect(self.calculate)

        self.data = None

    def signal_progress(self, value, maximum):
        self.update_progress.emit(value, maximum)

    @pyqtSlot(int, int)
    def set_progress(self, value, maximum):
        self.slice_progressBar.setMaximum(maximum)
        self.slice_progressBar.setValue(value)

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
            self.load_directory(dataFile)

    def loadNumpyArray(self, data):
        if np.max(data.flat) <= 10: data *= 100
        # print data.shape
        for sl in range(data.shape[2]):
            self.imList.append(data[:, :, sl])

    def load_dosma_volume(self, medical_volume):
        if np.max(medical_volume.volume) < 10:
            medical_volume *= 100
        while np.max(medical_volume.volume) > 10000:
            print(np.max(medical_volume.volume))
            medical_volume.volume /= 10
        self.medical_volume = medical_volume
        self.resolution = np.array(self.medical_volume.pixel_spacing)
        self.resolution_valid = True
        self.imList = ImListProxy(self.medical_volume)


    # load a whole directory of dicom files
    def loadDirectory_internal(self, path):
        self.imList = []
        self.resolution_valid = False
        self.resolution = [1, 1, 1]

        medical_volume, affine_valid, title, basepath, basename = dosma_volume_from_path(path, self)

        self.load_dosma_volume(medical_volume)
        self.resolution_valid = affine_valid
        self.data_location_Text.setText(basename)
        self.basename = basename
        self.basepath = basepath

        if len(self.imList) > 0:
            self.curImage = 0

    @separate_thread_decorator
    def load_directory(self, path, override_class=None):
        self.signal_progress(0, 1)
        self.imList = []
        self.mask_list = {}
        _, ext = os.path.splitext(path)
        mask_dictionary = None
        self.resolution = [1,1,1]
        self.resolution_valid = False
        if ext.lower() == '.npz':
            # data and mask bundle
            bundle = np.load(path, allow_pickle=False)
            if 'data' not in bundle:
                self.signal_alert('No data in bundle!')
                return
            if 'comment' in bundle:
                self.signal_alert('Loading bundle with comment:\n' + str(bundle['comment']))

            self.basepath = os.path.dirname(path)
            self.loadNumpyArray(bundle['data'])
            if 'resolution' in bundle:
                self.resolution = list(bundle['resolution'])
                self.resolution_valid = True
                print('Resolution', self.resolution)

            mask_dictionary = {}
            for key in bundle:
                if key.startswith('mask_'):
                    mask_name = key[len('mask_'):]
                    mask_dictionary[mask_name] = bundle[key]
                    print('Found mask', mask_name)
        else:
            self.loadDirectory_internal(self, path)

        # ask for resolution to be inserted
        if not self.resolution_valid:
            accepted, output = GenericInputDialog.show_dialog("Insert resolution", [
                GenericInputDialog.FloatSpinInput("X (mm)", 1, 0, 99, 0.1),
                GenericInputDialog.FloatSpinInput("Y (mm)", 1, 0, 99, 0.1),
                GenericInputDialog.FloatSpinInput("Slice (mm)", 1, 0, 99, 0.1)
            ], self.fig.canvas)
            if accepted:
                self.resolution = [output[0], output[1], output[2]]
                self.resolution_valid = True

        if mask_dictionary:
            self.setSplash(True, 1, 2, "Loading masks")
            self.mask_location_Text.setText(self.basename)
            self.masksToRois(mask_dictionary, 0)
        self.signal_progress(0, 1)

        if self.mask_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)

    def masksToRois(self, mask_dictionary):
        pass

    @pyqtSlot()
    def choose_mask_dir(self):
        maskDir = QFileDialog.getExistingDirectory(self, caption='Select folder containing other DICOM folders or Nifti files')

        if maskDir:
            self.mask_import(maskDir)

    @separate_thread_decorator
    def mask_import(self, filename):
        dicom_ext = ['.dcm', '.ima']
        nii_ext = ['.nii', '.gz']
        npy_ext = ['.npy']
        npz_ext = ['.npz']
        path = os.path.abspath(filename)
        _, ext = os.path.splitext(path)
        self.mask_location_Text.setText(path)
        self.mask_list = {}

        if os.path.isdir(path):
            containsDirs = False
            containsDicom = False
            nii_list = []
            dir_list = []
            firstDicom = None
            for element in os.listdir(path):
                if element.startswith('.'): continue
                new_path = os.path.join(path, element)
                if os.path.isdir(new_path):
                    containsDirs = True
                    dir_list.append(new_path)
                else:  # check if the folder contains dicoms
                    _, ext2 = os.path.splitext(new_path)
                    if ext2.lower() in dicom_ext:
                        containsDicom = True
                        if firstDicom is None:
                            firstDicom = new_path
                    elif ext2.lower() in nii_ext:
                        nii_list.append(new_path)

            if containsDicom and containsDirs:
                containsDicom = False

            if containsDicom:
                path = new_path  # "fake" the loading of the first image
                _, ext = os.path.splitext(path)
            elif containsDirs:
                ext = 'multidicom'  # "fake" extension to load a directory

        basename = os.path.basename(path)
        is3D = False

        self.signal_progress(0, 2)

        def fail(text):
            self.signal_progress(0, 2)
            self.signal_alert(text)

        def load_mask_validate(name, mask):
            if name.lower().endswith('.nii'):
                name = name[:-4]
            if mask.shape[0] != self.image.shape[0] or mask.shape[1] != self.image.shape[1]:
                print("Mask shape", mask.shape, "self image shape", self.image.shape)
                fail("Mask size mismatch")
                return
            if mask.ndim > 2:
                is3D = True
                if mask.shape[2] != len(self.imList):
                    print("Mask shape", mask.shape, "self length", len(self.imList))
                    fail("Mask size mismatch")
                    return
            mask = mask > 0
            self.masksToRois({name: mask})  # this is OK for 2D and 3D

        ext = ext.lower()

        if ext in npy_ext:
            mask = np.load(path)
            name = basename
            self.signal_progress(1, 2)
            load_mask_validate(name, mask)
            self.signal_progress(0,1)
            return
        if ext in npz_ext:
            mask_dict = np.load(path)
            n_masks = len(mask_dict)
            cur_mask = 0
            for name, mask in mask_dict.items():
                self.signal_progress(cur_mask, n_masks)
                load_mask_validate(name, mask)
            self.signal_progress(0,1)
            return
        elif ext in nii_ext:
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False)
            name, _ = os.path.splitext(os.path.basename(path))

            mask = mask_medical_volume.volume

            self.signal_progress(2, 3)
            load_mask_validate(name, mask)
            self.signal_progress(0, 1)
            return
        elif ext == 'multidicom' or len(nii_list) > 0:
            if ext == 'multidicom':
                path_list = dir_list
            else:
                path_list = nii_list
            # load multiple dicom masks and align them at the same time
            accumulated_mask = None
            current_mask_number = 1
            dicom_info_ok = None
            names = []
            for data_path in path_list:
                if data_path.startswith('.'): continue
                try:
                    mask_medical_volume, *_ = dosma_volume_from_path(data_path, reorient_data=False)
                except:
                    continue
                dataset = mask_medical_volume.volume
                dataset[dataset > 0] = 1
                dataset[dataset < 1] = 0
                name, _ = os.path.splitext(os.path.basename(data_path))

                self.signal_progress(2, 3)
                mask = dataset
                load_mask_validate(name, mask)
                self.signal_progress(0, 1)

        if self.data_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)


    # convert a single slice to ROIs
    def maskToRois2D(self, name, mask, imIndex):
        if np.any(mask):
            if imIndex not in self.mask_list:
                self.mask_list[imIndex] = {}
            self.mask_list[imIndex][name] = mask

    # convert a 2D mask or a 3D dataset to rois
    def masksToRois(self, maskDict):
        for name, mask in maskDict.items():
            if len(mask.shape) > 2:  # multislice
                for sl in range(mask.shape[2]):
                    self.maskToRois2D(name, mask[:, :, sl], sl)
            else:
                self.maskToRois2D(name, mask, 0)


def run():
    app = QApplication(sys.argv)
    window = QMainWindow()
    widget = BatchValidateWindow()
    window.setCentralWidget(widget)
    window.setWindowTitle("Batch validation")
    window.show()
    sys.exit(app.exec_())
