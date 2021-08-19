#  Copyright (c) 2021 Dafne-Imaging Team
import os
import sys
import traceback

import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QApplication, QMainWindow, QInputDialog

from config import config
from dl.RemoteModelProvider import RemoteModelProvider
from dl.misc import calc_dice_score
from ui import GenericInputDialog
from ui.ValidateUI import Ui_ValidateUI
from ui.pyDicomView import ImListProxy
from utils.ThreadHelpers import separate_thread_decorator
from utils.dicomUtils.misc import dosma_volume_from_path
import tensorflow as tf

CLASSIFICATION = 'Leg'
TIMESTAMP_INTERVAL = (1625471255, 1626096769)
UPLOAD_STATS = True


class BatchValidateWindow(QWidget, Ui_ValidateUI):
    update_progress = pyqtSignal(int, int, str)
    update_overall_progress = pyqtSignal(int, int)
    alert_signal = pyqtSignal(str)
    success = pyqtSignal()

    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("Batch model validation")
        self.update_progress.connect(self.set_progress)
        self.update_overall_progress.connect(self.set_overall_progress)
        self.alert_signal.connect(self.alert)
        self.data_choose_Button.clicked.connect(self.choose_data)
        self.mask_choose_Button.clicked.connect(self.choose_mask_dir)
        self.resolution = [1,1,1]
        self.resolution_valid = False
        self.medical_volume = None
        self.im_list = []
        self.basepath = ''
        self.basename = ''
        self.maskpath = ''
        self.mask_list = {}

        self.evaluate_Button.clicked.connect(self.start_calculation)

        self.data = None
        self.model_provider = None
        config.load_config()
        print("Timestamp interval", TIMESTAMP_INTERVAL)
        print("Logging enabled", UPLOAD_STATS)
        self.timestamps_to_download = []
        self.init_model_provider()

    def init_model_provider(self):
        self.model_provider = RemoteModelProvider(config.GlobalConfig['MODEL_PATH'], config.GlobalConfig['SERVER_URL'],
                                             config.GlobalConfig['API_KEY'], config.GlobalConfig['TEMP_UPLOAD_DIR'])

        try:
            available_models = self.model_provider.available_models()
        except:
            traceback.print_exc()
            self.signal_alert("Error connecting to server")
            sys.exit(-1)

        if CLASSIFICATION not in available_models:
            self.signal_alert("Model not found")
            sys.exit(-1)

        model_info = self.model_provider.model_details(CLASSIFICATION)
        timestamps = model_info['timestamps']
        self.timestamps_to_download = [timestamp for timestamp in timestamps if TIMESTAMP_INTERVAL[0] <= int(timestamp) <= TIMESTAMP_INTERVAL[1]]


        #self.model_provider.log("Testing")

        print("Connection established")

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
            self.data_location_Text.setText("")
            self.mask_location_Text.setText("")
            self.load_directory(dataFile)

    def loadNumpyArray(self, data):
        if np.max(data.flat) <= 10: data *= 100
        # print data.shape
        for sl in range(data.shape[2]):
            self.im_list.append(data[:, :, sl])

    def load_dosma_volume(self, medical_volume):
        if np.max(medical_volume.volume) < 10:
            medical_volume *= 100
        while np.max(medical_volume.volume) > 10000:
            print(np.max(medical_volume.volume))
            medical_volume.volume /= 10
        self.medical_volume = medical_volume
        self.resolution = np.array(self.medical_volume.pixel_spacing)
        self.resolution_valid = True
        self.im_list = ImListProxy(self.medical_volume)


    # load a whole directory of dicom files
    def loadDirectory_internal(self, path):
        self.im_list = []
        self.resolution_valid = False
        self.resolution = [1, 1, 1]

        medical_volume, affine_valid, title, basepath, basename = dosma_volume_from_path(path, self)

        self.load_dosma_volume(medical_volume)
        self.resolution_valid = affine_valid
        self.data_location_Text.setText(basename)
        self.basename = basename
        self.basepath = basepath

        if len(self.im_list) > 0:
            self.curImage = 0

    #@separate_thread_decorator
    def load_directory(self, path):
        self.data_choose_Button.setEnabled(False)
        self.signal_progress(0, 1, "Loading data")
        self.im_list = []
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
            self.loadDirectory_internal(path)


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

        self.data_location_Text.setText(self.basepath)
        self.mask_choose_Button.setEnabled(True)

        if mask_dictionary:
            self.setSplash(True, 1, 2, "Loading masks")
            self.mask_location_Text.setText(self.basename)
            self.masksToRois(mask_dictionary, 0)
            self.mask_choose_Button.setEnabled(False)
        self.signal_progress(0, 1)

        self.data_choose_Button.setEnabled(True)

        if self.mask_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)

        print('im_list len', len(self.im_list))

    def masksToRois(self, mask_dictionary):
        pass

    @pyqtSlot()
    def choose_mask_dir(self):
        maskDir = QFileDialog.getExistingDirectory(self, caption='Select folder containing other DICOM folders or Nifti files')

        if maskDir:
            self.mask_import(maskDir)

    @separate_thread_decorator
    def mask_import(self, filename):
        self.mask_choose_Button.setEnabled(False)
        self.data_choose_Button.setEnabled(False)
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

        self.signal_progress(0, 2, "Loading masks")

        def fail(text):
            self.signal_progress(0, 2)
            self.signal_alert(text)

        def load_mask_validate(name, mask):
            if name.lower().endswith('.nii'):
                name = name[:-4]
            if mask.shape[0] != self.im_list[0].shape[0] or mask.shape[1] != self.im_list[0].shape[1]:
                print("Mask shape", mask.shape, "self image shape", self.image.shape)
                fail("Mask size mismatch")
                return
            if mask.ndim > 2:
                is3D = True
                if mask.shape[2] != len(self.im_list):
                    print("Mask shape", mask.shape, "self length", len(self.im_list))
                    fail("Mask size mismatch")
                    return
            mask = mask > 0
            self.masksToRois({name: mask})  # this is OK for 2D and 3D

        ext = ext.lower()

        if ext in npy_ext:
            mask = np.load(path)
            name = basename
            self.signal_progress(1, 2, "Loading masks")
            load_mask_validate(name, mask)
            self.signal_progress(0,1)
        elif ext in npz_ext:
            mask_dict = np.load(path)
            n_masks = len(mask_dict)
            cur_mask = 0
            for name, mask in mask_dict.items():
                self.signal_progress(cur_mask, n_masks, "Loading masks")
                load_mask_validate(name, mask)
            self.signal_progress(0,1)
        elif ext in nii_ext:
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False)
            name, _ = os.path.splitext(os.path.basename(path))

            mask = mask_medical_volume.volume

            self.signal_progress(2, 3, "Loading masks")
            load_mask_validate(name, mask)
            self.signal_progress(0, 1)
        elif ext == 'multidicom' or len(nii_list) > 0:
            if ext == 'multidicom':
                path_list = dir_list
            else:
                path_list = nii_list
            # load multiple dicom masks and align them at the same time
            accumulated_mask = None
            current_mask_number = 0
            total_masks = len(path_list)
            dicom_info_ok = None
            names = []
            for data_path in path_list:
                self.signal_progress(current_mask_number, total_masks, "Loading masks")
                current_mask_number += 1
                if data_path.startswith('.'): continue
                try:
                    mask_medical_volume, *_ = dosma_volume_from_path(data_path, reorient_data=False)
                except:
                    continue
                dataset = mask_medical_volume.volume
                dataset[dataset > 0] = 1
                dataset[dataset < 1] = 0
                name, _ = os.path.splitext(os.path.basename(data_path))

                self.signal_progress(2, 3, "Loading masks")
                mask = dataset
                load_mask_validate(name, mask)
                self.signal_progress(0, 1)

        if self.data_location_Text.text() != '':
            self.evaluate_Button.setEnabled(True)
        else:
            self.evaluate_Button.setEnabled(False)

        self.mask_choose_Button.setEnabled(True)
        self.data_choose_Button.setEnabled(True)
        self.signal_progress(0, 2)
        print("Masks:")
        for key in self.mask_list:
            print(key, ", ".join(self.mask_list[key].keys()))


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

    @pyqtSlot()
    def start_calculation(self):
        n_slices = len(self.mask_list)
        n_rois = len(next(iter(self.mask_list.values()))) # get the number of rois from an arbitrary slice
        accept = QMessageBox.question(self, "Run validation?", f'Running validation for model {CLASSIFICATION} on {len(self.timestamps_to_download)} models\nwith {n_rois} ROIs over {n_slices} slices. Continue?', QMessageBox.Yes | QMessageBox.No)
        if accept == QMessageBox.No:
            return
        comment, ok = QInputDialog.getText(self, "Add comment", "Add an identifier for this dataset")
        if not ok:
            return
        self.setEnabled(False)
        self.calculate(comment)

    @separate_thread_decorator
    def calculate(self, comment):
        split_laterality = False
        print("Imlist len", len(self.im_list))
        print("Masks:")
        for key in self.mask_list:
            print(key, ", ".join(self.mask_list[key].keys()))
            for roi_name in self.mask_list[key]:
                if roi_name.endswith('_L'):
                    split_laterality = True

        print("Split laterality:", split_laterality)
        print("Models to download", self.timestamps_to_download)

        n_models = len(self.timestamps_to_download)
        current_model_n = 0

        for timestamp in self.timestamps_to_download:
            self.signal_overall_progress(current_model_n, n_models)

            model = self.model_provider.load_model(CLASSIFICATION,
                                                  lambda value, maximum: self.signal_progress(value, maximum, f'Downloading {timestamp}'),
                                                  timestamp=timestamp)

            # perform segmentation
            n_slices = len(self.mask_list)
            current_slice = 0
            dice_scores = []
            n_voxels = []
            for slice, mask_dict in self.mask_list.items():
                self.signal_progress(current_slice, n_slices, f'Evaluating {timestamp}')
                input_dict = {'image': self.im_list[slice],
                              'classification': CLASSIFICATION,
                              'resolution': self.resolution[0:2],
                              'split_laterality': split_laterality}

                output_masks = model.apply(input_dict)
                for mask_name, mask in mask_dict.items():
                    if mask_name in output_masks:
                        n_vox = mask.sum()
                        dice = calc_dice_score(mask, output_masks[mask_name])
                        n_voxels.append(n_vox)
                        dice_scores.append(dice)
                        #print('Slice', slice, 'Mask', mask_name, 'N', n_vox, 'Dice', dice)
                current_slice += 1

            dice_scores = np.array(dice_scores)
            n_voxels = np.array(n_voxels)
            # print(diceScores)
            if np.sum(n_voxels) == 0:
                average_dice = -1.0
            else:
                average_dice = np.average(dice_scores, weights=n_voxels)

            message = f'Average dice - {comment} - model {timestamp}: {average_dice}'
            print(message)
            if UPLOAD_STATS:
                self.model_provider.log(message)

            current_model_n += 1

            # free memory
            del model
            try:
                tf.keras.backend.clear_session() # this should clear the memory leaks by tensorflow
            except:
                print("Error cleaning keras session")

        self.signal_overall_progress(0,1)
        self.signal_progress(0,1,'Finished')
        self.setEnabled(True)


def run():
    app = QApplication(sys.argv)
    window = QMainWindow()
    widget = BatchValidateWindow()
    window.setCentralWidget(widget)
    window.setWindowTitle("Batch validation")
    window.show()
    sys.exit(app.exec_())
