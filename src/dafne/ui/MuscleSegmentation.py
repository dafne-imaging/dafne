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
import gc
import re
from urllib.error import URLError

import flexidep
import os, time, math, sys

from dafne_dl.model_loaders import ensure_compatible_orientation_inplace, ensure_compatible_orientation
import dafne_sam2.public_api as sam_api

from ..config import GlobalConfig, load_config
load_config()

import tensorflow as tf

if GlobalConfig['USE_GPU_FOR'] == 'Torch':
    # force CPU for tensorflow
    tf.config.set_visible_devices([], 'GPU')
    
import torch
if GlobalConfig['USE_GPU_FOR'] == 'Both (careful!)':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GlobalConfig['TENSORFLOW_MEMORY_ALLOCATION']*1000)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def determine_sam_device():
    if GlobalConfig['USE_GPU_FOR'] != 'Tensorflow':
        if torch.cuda.is_available():
            print('SAM loaded on GPU')
            return 'cuda'
        else:
            # should use GPU but we don't know what. Let SAM decide
            return 'auto'
    # force GPU only for tensorflow, SAM will use CPU
    print('SAM loaded on CPU')
    return 'cpu'

import matplotlib
from dafne_dl.common.biascorrection import biascorrection_image
from matplotlib.patches import Rectangle
from voxel import NiftiWriter, MedicalVolume
from voxel.orientation import to_RAS_affine
from scipy.interpolate import interp1d
from skimage.morphology import area_opening, area_closing

from .WhatsNew import NewsChecker, WhatsNewDialog
from dicomUtils.misc import realign_medical_volume, dosma_volume_from_path, reorient_data_ui, \
    get_nifti_orientation
from . import GenericInputDialog, hue_compass_colormap
from .DimensionSelectionDialog import reduce_array_dimensions
from ..utils.mask_to_spline import mask_average, mask_to_trivial_splines, masks_splines_to_splines_masks
from ..utils.pySplineInterp import SplineInterpROIClass
from ..utils.resource_utils import get_resource_path

matplotlib.use("Qt5Agg")

from .ToolboxWindow import ToolboxWindow
from dicomUtils.ui.pyDicomView import ImageShow, ImListProxy
from ..utils.mask_utils import save_npy_masks, save_npz_masks, save_dicom_masks, save_nifti_masks, \
    save_single_dicom_dataset, save_single_nifti, save_nifti_masks_3D
from dafne_dl.misc import get_model_detail, calc_dice_score_3D, calc_dice_score
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import shutil
from datetime import datetime
from ..utils.ROIManager import ROIManager
from .. import utils
sys.modules['utils'] = utils # to make pickle work

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from ..utils import compressed_pickle as pickle
import os.path
from collections import deque, OrderedDict
import functools
import csv

from ..utils.ThreadHelpers import separate_thread_decorator, main_thread_dialog_runner

from .BrushPatches import SquareBrush, PixelatedCircleBrush
from .ContourPainter import ContourPainter
import traceback

from dafne_dl.LocalModelProvider import LocalModelProvider
from dafne_dl.RemoteModelProvider import RemoteModelProvider
from dafne_dl.MixedModelProvider import MixedModelProvider

from ..utils.RegistrationManager import RegistrationManager

import requests

try:
    import SimpleITK as sitk # this requires simpleelastix!
except:
    sitk = None

try:
    import radiomics
except:
    radiomics = None

import subprocess

if os.name == 'posix':
    def checkCapsLock():
        return (int(subprocess.check_output('xset q | grep LED', shell=True)[65]) & 1) == 1
elif os.name == 'nt':
    import ctypes

    hllDll = ctypes.WinDLL("User32.dll")


    def checkCapsLock():
        return ((hllDll.GetKeyState(0x14) & 1) == 1)
else:
    def checkCapsLock():
        return False

try:
    QString("")
except:
    def QString(s):
        return s


INTENSITY_AWARE_THRESHOLD = 0.5
ACTIONS_TO_REMOVE = 'Subplots', 'Customize', 'Save'

def make_excepthook(muscle_segmentation_instance):
    def excepthook(exctype, value, traceback):
        muscle_segmentation_instance.alert(f"An error occurred. Please check the logs in {os.path.dirname(GlobalConfig['ERROR_LOG_FILE'])} for more information. The current ROIs will be saved.")
        muscle_segmentation_instance.saveROIPickle()
        muscle_segmentation_instance.close_slot()
        return sys.__excepthook__(exctype, value, traceback)
    return excepthook

def makeMaskLayerColormap(color):
    return matplotlib.colors.ListedColormap(np.array([
        [0, 0, 0, 0],
        [*color[:3],1]]))


def snapshotSaver(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.saveSnapshot()
        func(self, *args, **kwargs)

    return wrapper


def timeSnapshotSaver(func):
    # snapshot the ROIs of every time frame: for operations that modify multiple frames
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.saveSnapshot(all_timepoints=True)
        func(self, *args, **kwargs)

    return wrapper


class MuscleSegmentation(ImageShow, QObject):

    undo_possible = pyqtSignal(bool)
    redo_possible = pyqtSignal(bool)
    splash_signal = pyqtSignal(bool, int, int, str)
    reblit_signal = pyqtSignal()
    redraw_signal = pyqtSignal()
    reduce_brush_size = pyqtSignal()
    increase_brush_size = pyqtSignal()
    alert_signal = pyqtSignal(str, str)
    undo_signal = pyqtSignal()
    redo_signal = pyqtSignal()

    mask_changed = pyqtSignal(list, np.ndarray)
    mask_slice_changed = pyqtSignal(int, np.ndarray)
    volume_loaded_signal = pyqtSignal(list, np.ndarray)
    other_mask_changed = pyqtSignal(list, np.ndarray)
    displayed_slice_changed = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        self.suppressRedraw = False
        ImageShow.__init__(self, *args, **kwargs)
        QObject.__init__(self)

        self.shortcuts = {
            'z': self.undo_signal.emit,
            'y': self.redo_signal.emit,
            'g': self.gotoImageDialog
        }

        if GlobalConfig['CHECK_UPDATES']:
            try:
                self.check_updates()
            except URLError:
                print("Could not check for updates. No internet connection?")

        self.news_checker = NewsChecker()
        self.news_checker.news_ready.connect(self.show_news)
        self.news_checker.check_news()


        self.fig.canvas.mpl_connect('close_event', self.closeCB)
        # self.instructions = "Shift+click: add point, Shift+dblclick: optimize/simplify, Ctrl+click: remove point, Ctrl+dblclick: delete ROI, n: propagate fw, b: propagate back"

        if 'Elastix' in dir(sitk):
            self.registration_available = True
        else:
            print("Elastix is not available")
            self.registration_available = False

        self.app = None

        self.setupToolbar()

        main_window = self.fig.canvas.parent()
        main_window.setWindowTitle("Dafne Main Window")
        with get_resource_path('dafne_logo.png') as logo_path:
            main_window.setWindowIcon(QIcon(logo_path))

        self.roiManager = None

        self.wacom = False

        self.saveDicom = False

        self.model_provider = None
        self.dl_classifier = None
        self.dl_segmenters = {}
        self.model_details = {}

        # self.fig.canvas.setCursor(Qt.BlankCursor)

        # self.setCmap('viridis')
        self.extraOutputParams = []

        self.registrationManager = None

        self.hideRois = False
        self.editMode = ToolboxWindow.EDITMODE_MASK
        self.resetInternalState()

        self.fig.canvas.mpl_connect('resize_event', self.resizeCB)
        self.reblit_signal.connect(self.do_reblit)
        self.redraw_signal.connect(self.do_redraw)
        self.undo_signal.connect(self.undo)
        self.redo_signal.connect(self.redo)

        self.separate_thread_running = False

        toolbar = self.fig.canvas.toolbar
        actions = toolbar.actions()
        for action in actions:
            if action.text() in ACTIONS_TO_REMOVE:
                toolbar.removeAction(action)


        # disable keymapping from matplotlib - avoid pan and zoom
        for key in list(plt.rcParams):
            if 'keymap' in key and 'zoom' not in key and 'pan' not in key:
                plt.rcParams[key] = []
        sys.excepthook = make_excepthook(self)

        # 3D incremental learning
        self.incrLearnDataTrain = {}
        self.incrLearnSegTrain = {}
        self.incrementalLearningAffine = {}
        self.incrLearnMeanDice = {}
        self.bundle_saved_for_IL = False

        # SAM
        self.sam = None

    def get_sam(self):
        if self.sam is not None:
            return self.sam

        def set_progress(progress, total):
            self.setSplash(True, progress, total, "Loading SAM model...")

        sam_model = GlobalConfig['SAM_MODEL']
        checkpoint_dir = GlobalConfig['MODEL_PATH']

        self.sam = sam_api.load_segmenter(checkpoint_dir, sam_model, device=determine_sam_device(), progress_callback=set_progress)
        return self.sam

    @pyqtSlot(list, str)
    def show_news(self, news, index_address):
        d = WhatsNewDialog(news, index_address)
        d.exec()

    def check_updates(self):
        versions = flexidep.get_installed_packages_with_available_versions(['dafne', 'dafne-dl'])
        if not (versions['dafne-dl']['latest'] and versions['dafne-dl']['latest']):
            # either dafne or dafne-dl are not latest versions
            answer = QMessageBox.question(None, 'Update available',
                                          'An update is available. Do you want to update dafne?',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer == QMessageBox.Yes:
                flexidep.install_package(flexidep.PackageManagers.pip, 'dafne-dl', extra_command_line='-U')
                flexidep.install_package(flexidep.PackageManagers.pip, 'dafne', extra_command_line='-U')
                QMessageBox.information(None, 'Update complete', 'Dafne was updated. Please restart the application to use the new version.')




    def get_app(self):
        if not self.app:
            self.app = QApplication.instance()
        return self.app


    def resizeCB(self, event):
        self.resetBlitBg()
        self.redraw()

    def resetBlitBg(self):
        self.blitBg = None


    @pyqtSlot()
    def resetModelProvider(self):
        available_models = None
        filter_classes = False
        self.model_details = {}
        if GlobalConfig['MODEL_PROVIDER'] == 'Local':
            model_provider = LocalModelProvider(GlobalConfig['MODEL_PATH'], GlobalConfig['TEMP_UPLOAD_DIR'])
            available_models = model_provider.available_models()
        else:
            if GlobalConfig['MODEL_PROVIDER'] == 'Remote':
                ProviderClass = RemoteModelProvider
            else:
                ProviderClass = MixedModelProvider
                print('Using mixed model provider')
            url = GlobalConfig['SERVER_URL']
            if not url.endswith('/'):
                url += '/'
            model_provider = ProviderClass(GlobalConfig['MODEL_PATH'], url, GlobalConfig['API_KEY'], GlobalConfig['TEMP_UPLOAD_DIR'])
            fallback = False
            try:
                available_models = model_provider.available_models()
                filter_classes = True
            except PermissionError:
                self.alert("Error in using Remote Model. Please check your API key. Falling back to Local")
                fallback = True
            except requests.exceptions.ConnectionError:
                self.alert("Remote server unavailable. Falling back to Local")
                fallback = True
            except requests.exceptions.InvalidURL:
                self.alert("Invalid URL. Falling back to Local")
                fallback = True
            else:
                if available_models is None:
                    self.alert("Error in using Remote Model Loading. Falling back to Local")
                    fallback = True

            if fallback:
                GlobalConfig['MODEL_PROVIDER'] = 'Local'
                model_provider = LocalModelProvider(GlobalConfig['MODEL_PATH'], GlobalConfig['TEMP_UPLOAD_DIR'])
                filter_classes = False
                available_models = model_provider.available_models()

        try:
            local_model_list = self.model_provider.get_local_models()
        except AttributeError:
            local_model_list = []

        GlobalConfig['ENABLED_MODELS'].extend(local_model_list)
        self.setModelProvider(model_provider)

        print(available_models)
        self.setAvailableClasses(available_models, filter_classes)

    @pyqtSlot()
    def configChanged(self):
        self.scroll_debounce_time = float(GlobalConfig['MOUSE_SCROLL_DEBOUNCE_TIME'])/1000.0
        self.resetInterface()
        self.resetModelProvider()

    def resetInterface(self):
        self.blitBg = None
        self.blitXlim = None
        self.blitYlim = None
        try:
            self.brush_patch.remove()
        except:
            pass

        self.brush_patch = None

        try:
            self.removeMasks()
        except:
            pass

        try:
            self.removeSubregion()
        except:
            pass

        self.maskImPlot = None
        self.maskOtherImPlot = None
        self.activeMask = None
        self.otherMask = None
        self.region_rectangle = None

        self.roiColor = GlobalConfig['ROI_COLOR']
        self.roiOther = GlobalConfig['ROI_OTHER_COLOR']
        self.roiSame = GlobalConfig['ROI_SAME_COLOR']
        self.interpolation = GlobalConfig['INTERPOLATION']
        try:
            self.imPlot.set_interpolation(self.interpolation)
        except:
            pass

        self.setCmap(GlobalConfig['COLORMAP'])

        self.mask_layer_colormap = makeMaskLayerColormap(self.roiColor)
        self.mask_layer_other_colormap = makeMaskLayerColormap(self.roiOther)

        try:
            self.removeContours()
        except:
            pass

        self.activeRoiPainter = ContourPainter(self.roiColor, GlobalConfig['ROI_CIRCLE_SIZE'])
        self.sameRoiPainter = ContourPainter(self.roiSame, 0.1)
        self.otherRoiPainter = ContourPainter(self.roiOther, 0.1)

        try:
            self.updateContourPainters()
        except:
            pass

        self.redraw()

    def resetInternalState(self):
        self.imList = []
        self.resolution = [1, 1, 1]
        self.curImage = 0
        self.classifications = []
        self.lastsave = datetime.now()

        self.roiChanged = {}
        self.history = deque(maxlen=GlobalConfig['HISTORY_LENGTH'])
        self.historyHead = None
        self.currentHistoryPoint = 0

        self.registrationManager = None

        self.resetModelProvider()
        self.resetInterface()
        self.slicesUsedForTraining = set()

        self.roiManager = None

        self.currentPoint = None
        self.translateDelta = None
        self.rotationDelta = None
        self.scroll_debounce_time = float(GlobalConfig['MOUSE_SCROLL_DEBOUNCE_TIME'])/1000.0
        self.threshold_mask = None

        self.bundle_saved_for_IL = False

        self.additional_contrasts = OrderedDict()
        self.additional_contrast_frames = {} # name -> list of per-timepoint volumes (time-resolved contrasts)
        self.current_contrast = ToolboxWindow.BASE_CONTRAST_LABEL
        self.toolbox_window.clear_contrast_combo()

        # time-resolved (4D) datasets: one 3D MedicalVolume per time frame.
        # Empty list means the dataset has no time component.
        self.time_frames = []
        self.current_timepoint = 0
        self.roiManagers = {}
        self.registrationManagers = {}
        self.toolbox_window.set_timepoints(1)


    #############################################################################################
    ###
    ### Toolbar interaction
    ###
    ##############################################################################################

    def setupToolbar(self):
        self.toolbox_window = ToolboxWindow(self, activate_registration=self.registration_available, activate_radiomics= (radiomics is not None))
        self.toolbox_window.show()

        self.toolbox_window.editmode_changed.connect(self.changeEditMode)

        self.toolbox_window.roi_added.connect(self.addRoi)
        self.toolbox_window.subroi_added.connect(self.addSubRoi)

        self.toolbox_window.roi_deleted.connect(self.removeRoi)
        self.toolbox_window.subroi_deleted.connect(self.removeSubRoi)

        self.toolbox_window.roi_changed.connect(self.changeRoi)

        self.toolbox_window.roi_clear.connect(self.clearCurrentROI)

        self.toolbox_window.do_autosegment.connect(self.doSegmentationMultislice)

        self.toolbox_window.classification_changed.connect(self.changeClassification)
        self.toolbox_window.classification_change_all.connect(self.changeAllClassifications)

        self.toolbox_window.undo.connect(self.undo)
        self.toolbox_window.redo.connect(self.redo)
        self.undo_possible.connect(self.toolbox_window.undo_enable)
        self.redo_possible.connect(self.toolbox_window.redo_enable)

        self.toolbox_window.contour_simplify.connect(self.simplify)
        self.toolbox_window.contour_optimize.connect(self.optimize)

        self.toolbox_window.calculate_transforms.connect(self.calcTransforms)
        self.toolbox_window.contour_propagate_fw.connect(self.propagate)
        self.toolbox_window.contour_propagate_bw.connect(self.propagateBack)

        self.toolbox_window.interpolate_mask.connect(self.interpolate)
        self.toolbox_window.interpolate_block.connect(self.interpolate_block)

        self.toolbox_window.roi_import.connect(self.loadROIPickle)
        self.toolbox_window.roi_export.connect(self.saveROIPickle)
        self.toolbox_window.data_open.connect(self.loadDirectory)
        self.toolbox_window.data_save_as_nifti.connect(self.save_data_as_reoriented_nifti)
        self.toolbox_window.data_reorient.connect(self.reorient_data)
        self.toolbox_window.masks_export.connect(self.saveResults)
        self.toolbox_window.bundle_export.connect(self.saveBundle)

        self.toolbox_window.roi_copy.connect(self.copyRoi)
        self.toolbox_window.roi_combine.connect(self.combineRoi)
        self.toolbox_window.roi_multi_combine.connect(self.combineMultiRoi)
        self.toolbox_window.roi_remove_overlap.connect(self.roiRemoveOverlap)

        self.toolbox_window.statistics_calc.connect(self.saveStats)
        self.toolbox_window.statistics_calc_slicewise.connect(self.saveStats_singleslice)
        self.toolbox_window.radiomics_calc.connect(self.saveRadiomics)

        self.toolbox_window.incremental_learn.connect(self.incrementalLearnStandalone)

        self.toolbox_window.mask_import.connect(self.loadMask)

        self.splash_signal.connect(self.toolbox_window.set_splash)
        self.interface_disabled = False
        self.splash_signal.connect(self.disableInterface)

        self.toolbox_window.mask_grow.connect(self.maskGrow)
        self.toolbox_window.mask_shrink.connect(self.maskShrink)
        self.toolbox_window.mask_fill_holes.connect(self.maskFillHoles)
        self.toolbox_window.mask_despeckle.connect(self.maskDespeckle)
        self.toolbox_window.mask_auto_threshold.connect(self.maskAutoThreshold)
        self.toolbox_window.sam_autorefine.connect(self.samAutoRefine)

        self.toolbox_window.config_changed.connect(self.configChanged)
        self.toolbox_window.data_upload.connect(self.uploadData)

        self.toolbox_window.model_import.connect(self.importModel)

        self.reduce_brush_size.connect(self.toolbox_window.reduce_brush_size)
        self.increase_brush_size.connect(self.toolbox_window.increase_brush_size)
        self.toolbox_window.brush_changed.connect(self.updateBrush)

        self.alert_signal.connect(self.toolbox_window.alert)
        self.toolbox_window.quit.connect(self.close_slot)

        self.toolbox_window.reblit.connect(self.do_reblit)
        self.toolbox_window.redraw.connect(self.do_redraw)

        self.toolbox_window.delete_subregion.connect(self.delete_current_subregion)
        self.toolbox_window.delete_all_subregions.connect(self.delete_all_subregions)
        self.toolbox_window.copy_all_subregions.connect(self.copy_all_subregions)

        self.toolbox_window.mask_transfer.connect(self.transfer_roi)

        self.toolbox_window.show_3D_viewer_signal.connect(self.emit_viewer3d_data)
        self.mask_changed.connect(self.toolbox_window.viewer3D.set_spacing_and_data)
        self.mask_slice_changed.connect(self.toolbox_window.viewer3D.set_slice)
        self.volume_loaded_signal.connect(self.toolbox_window.viewer3D.set_spacing_and_anatomy)
        self.other_mask_changed.connect(self.toolbox_window.viewer3D.set_other_masks)
        self.displayed_slice_changed.connect(self.toolbox_window.viewer3D.set_main_slice)
        self.toolbox_window.viewer3D.main_slice_changed.connect(self.viewer3d_slice_changed)

        self.toolbox_window.data_add.connect(self.load_additional_contrast)
        self.toolbox_window.contrast_changed.connect(self.change_contrast)
        self.toolbox_window.delete_contrast.connect(self.delete_additional_contrast)

        self.toolbox_window.timepoint_changed.connect(self.change_timepoint)
        self.toolbox_window.time_copy.connect(self.time_copy)
        self.toolbox_window.time_interpolate.connect(self.time_interpolate)
        self.toolbox_window.time_interpolate_block.connect(self.time_interpolate_block)

    def setSplash(self, is_splash, current_value = 0, maximum_value = 1, text= ""):
        #print("setSplash", is_splash, current_value, maximum_value, text)
        self.splash_signal.emit(is_splash, current_value, maximum_value, text)

    #dis/enable interface callbacks
    @pyqtSlot(bool, int, int, str)
    def disableInterface(self, disable, unused1, unused2, txt):
        if self.interface_disabled == disable: return
        self.interface_disabled = disable
        if disable:
            self.disconnectSignals()
        else:
            self.connectSignals()

    @pyqtSlot(str)
    def changeEditMode(self, mode):
        print("Changing edit mode")
        self.setSplash(True, 0, 1)
        self.editMode = mode
        roi_name = self.getCurrentROIName()
        if roi_name:
            self.updateRoiList()
            self.toolbox_window.set_current_roi(roi_name)
            if mode == ToolboxWindow.EDITMODE_MASK:
                self.removeContours()
                self.updateMasksFromROIs()
            else:
                self.removeMasks()
                self.updateContourPainters()
            self.redraw()
        self.setSplash(False, 1, 1)

    def setState(self, state):
        self.state = state

    def getState(self):
        if self.toolbox_window.valid_roi(): return 'MUSCLE'
        return 'INACTIVE'

    def updateRoiList(self):
        if not self.roiManager: return
        roiDict = {}
        imageN = int(self.curImage)
        for roiName in self.roiManager.get_roi_names():
            if self.editMode == ToolboxWindow.EDITMODE_MASK:
                if not self.roiManager.contains(roiName, imageN):
                    self.roiManager.add_mask(roiName, imageN)
                n_subrois = 1
            else:
                if not self.roiManager.contains(roiName, imageN) or self.roiManager.get_roi_mask_pair(roiName,
                                                                                                      imageN).get_subroi_len() == 0:
                    self._addSubRoi_internal(roiName, imageN)
                n_subrois = self.roiManager.get_roi_mask_pair(roiName, imageN).get_subroi_len()
            roiDict[roiName] = n_subrois  # dict: roiname -> n subrois per slice
        self.toolbox_window.set_rois_list(roiDict)
        self.updateContourPainters()
        self.updateMasksFromROIs()

    def alert(self, text, type="Warning"):
        self.alert_signal.emit(text, type)

    def question(self, text, question_type="YesNo"):
        return self.toolbox_window.question(text, question_type)

    #############################################################################################
    ###
    ### History
    ###
    #############################################################################################

    def saveSnapshot(self, save_head = False, all_timepoints = False):
        #print("Saving snapshot")
        if self.roiManager is None:
            try:
                self.roiManager = ROIManager(self.imList[0].shape)
            except:
                return
        # the head snapshot must always capture every frame, because the state being redone
        # to might have been saved by a multi-frame operation
        if (all_timepoints or save_head) and self.has_time_dimension():
            current_point = pickle.dumps({'timepoint': self.current_timepoint, 'roiManagers': self.roiManagers})
        else:
            current_point = pickle.dumps({'timepoint': self.current_timepoint, 'roiManager': self.roiManager})
        if save_head:
            #print("Saving head state")
            self.historyHead = current_point
        else:
            # clear history until the current point, so we can't redo anymore
            while self.currentHistoryPoint > 0:
                self.history.popleft()
                self.currentHistoryPoint -= 1
            self.history.appendleft(current_point)
            self.historyHead = None

        self.undo_possible.emit(self.canUndo())
        self.redo_possible.emit(self.canRedo())

    def canUndo(self):
        #print("Can undo history point", self.currentHistoryPoint, "len history", len(self.history))
        return self.currentHistoryPoint < len(self.history)

    def canRedo(self):
        return self.currentHistoryPoint > 0 or self.historyHead is not None

    def _changeHistory(self):
        #print('Current history point', self.currentHistoryPoint, 'history len', len(self.history))
        if self.currentHistoryPoint == 0 and self.historyHead is None:
            print('Warning: invalid redo')
            return
        roiName = self.getCurrentROIName()
        subRoiNumber = self.getCurrentSubroiNumber()
        if self.currentHistoryPoint == 0:
            #print("loading head")
            saved_state = pickle.loads(self.historyHead)
            self.historyHead = None
        else:
            #print("loading", self.currentHistoryPoint-1)
            saved_state = pickle.loads(self.history[self.currentHistoryPoint-1])

        restored_managers = None
        if isinstance(saved_state, dict):
            saved_timepoint = saved_state.get('timepoint', 0)
            restored_managers = saved_state.get('roiManagers', None) # multi-frame snapshot
            if restored_managers is not None:
                restored_manager = restored_managers[saved_timepoint]
            else:
                restored_manager = saved_state['roiManager']
        else: # old-style snapshot containing the bare ROIManager
            saved_timepoint = self.current_timepoint
            restored_manager = saved_state

        if self.has_time_dimension() and saved_timepoint != self.current_timepoint:
            # the snapshot was taken on another time frame: jump back to it
            self.toolbox_window.set_current_timepoint(saved_timepoint)

        self.clearAllROIs()
        if restored_managers is not None and self.has_time_dimension():
            self.roiManagers = restored_managers
            self.roiManager = restored_managers[self.current_timepoint]
        else:
            self.roiManager = restored_manager
            if self.roiManagers:
                self.roiManagers[self.current_timepoint] = restored_manager

        self.updateRoiList()
        if self.roiManager.contains(roiName):
            if self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK:
                self.toolbox_window.set_current_roi(roiName, -1)
            else:
                if subRoiNumber < self.roiManager.get_roi_mask_pair(roiName, self.curImage).get_subroi_len():
                    self.toolbox_window.set_current_roi(roiName, subRoiNumber)
                else:
                    self.toolbox_window.set_current_roi(roiName, 0)
        self.activeMask = None
        self.otherMask = None
        self.redraw()
        self.undo_possible.emit(self.canUndo())
        self.redo_possible.emit(self.canRedo())

    @pyqtSlot()
    def undo(self):
        if not self.canUndo(): return
        if self.currentHistoryPoint == 0:
            self.saveSnapshot(save_head=True)  # push current status into the history for redo
        self.currentHistoryPoint += 1
        self._changeHistory()

    @pyqtSlot()
    def redo(self):
        if not self.canRedo(): return
        self.currentHistoryPoint -= 1
        self._changeHistory()

    ############################################################################################################
    ###
    ### ROI management
    ###
    #############################################################################################################

    def getRoiFileName(self):
        if self.basename:
            roi_fname = self.basename + '.' + GlobalConfig['ROI_FILENAME']
        else:
            roi_fname = GlobalConfig['ROI_FILENAME']
        return os.path.join(self.basepath, roi_fname)

    def clearAllROIs(self):
        self.roiManager.clear()
        self.updateRoiList()
        self.reblit()

    def clearSubrois(self, name, sliceN):
        self.roiManager.clear(name, sliceN)
        self.updateRoiList()
        self.reblit()

    @pyqtSlot(str)
    @snapshotSaver
    def removeRoi(self, roi_name):
        self.roiManager.clear(roi_name)
        self.updateRoiList()
        self.reblit()

    @pyqtSlot(int)
    @snapshotSaver
    def removeSubRoi(self, subroi_number):
        current_name, _ = self.toolbox_window.get_current_roi_subroi()
        self.roiManager.clear_subroi(current_name, int(self.curImage), subroi_number)
        self.updateRoiList()
        self.reblit()

    @pyqtSlot(str)
    @snapshotSaver
    def addRoi(self, roiName):
        if self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.roiManager.add_mask(roiName, int(self.curImage))
        else:
            self.roiManager.add_roi(roiName, int(self.curImage))
        self.updateRoiList()
        self.toolbox_window.set_current_roi(roiName, 0)
        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.reblit()
        self.emit_mask_changed()

    def _addSubRoi_internal(self, roi_name=None, imageN=None):
        if not roi_name:
            roi_name, _ = self.toolbox_window.get_current_roi_subroi()
        if imageN is None:
            imageN = int(self.curImage)
        self.roiManager.add_subroi(roi_name, imageN)

    @pyqtSlot()
    #@snapshotSaver this generates too many calls; anyway we want to add the subroi to the history
    # when something happens to it
    def addSubRoi(self, roi_name=None, imageN=None):
        if not roi_name:
            roi_name, _ = self.toolbox_window.get_current_roi_subroi()
        if imageN is None:
            imageN = int(self.curImage)
        self._addSubRoi_internal(roi_name, imageN)
        self.updateRoiList()
        self.toolbox_window.set_current_roi(roi_name, self.roiManager.get_roi_mask_pair(roi_name,
                                                                                        imageN).get_subroi_len() - 1)
        self.reblit()

    @pyqtSlot(str, int)
    def changeRoi(self, roi_name, subroi_index):
        """ Change the active ROI """
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
        self.reblit()
        self.emit_mask_changed()

    def getCurrentROIName(self):
        """ Gets the name of the ROI selected in the toolbox """
        return self.toolbox_window.get_current_roi_subroi()[0]

    def getCurrentSubroiNumber(self):
        return self.toolbox_window.get_current_roi_subroi()[1]

    def _getSetCurrentROI(self, offset=0, newROI=None):
        """ Generic get/set for ROI objects inside the roi manager """
        if not self.getCurrentROIName():
            return None

        imageN = int(self.curImage + offset)
        curName = self.getCurrentROIName()
        curSubroi = self.getCurrentSubroiNumber()

        #print("Get set ROI", curName, imageN, curSubroi)

        return self.roiManager._get_set_roi(curName, imageN, curSubroi, newROI)

    def getCurrentROI(self, offset=0):
        """ Get current ROI object """
        return self._getSetCurrentROI(offset)

    def setCurrentROI(self, r, offset=0):
        self._getSetCurrentROI(offset, r)

    def getCurrentMask(self, offset=0):
        roi_name = self.getCurrentROIName()
        if not self.roiManager or not roi_name:
            return None
        return self.roiManager.get_mask(roi_name, int(self.curImage + offset))

    def setCurrentMask(self, mask, offset=0):
        roi_name = self.getCurrentROIName()
        if not self.roiManager or not roi_name:
            return None
        self.roiManager.set_mask(roi_name, int(self.curImage + offset), mask)

    def getCurrentSubregion(self, offset=0):
        if not self.roiManager:
            return None
        imageN = int(self.curImage + offset)
        return self.roiManager.get_autosegment_subregion(imageN)

    def setCurrentSubregion(self, subregion, offset=0):
        if not self.roiManager:
            return
        imageN = int(self.curImage + offset)
        if subregion is None:
            self.roiManager.clear_autosegment_subregion(imageN)
        else:
            self.roiManager.set_autosegment_subregion(imageN, subregion)
    
    def map_orientation(self, orientation):
        """Orientation map AP, RL, IS"""
        mapping = {
            'A': 'AP', 'P': 'AP',  
            'R': 'RL', 'L': 'RL',  
            'S': 'IS', 'I': 'IS'   
        }
        
        return tuple(mapping[char] for char in orientation)

    def process_orientation(self, orientation, segm):
        """Verify category of orientation"""

        def transpose_mask(case, segm):
            """Transpose the stack based on the orientation."""
            if case == "Case axial: AP-RL-IS":
                mask = np.transpose(np.stack(segm), [1, 0, 2])

            elif case == "Case sagittal: IS-AP-RL":
                mask = np.transpose(np.stack(segm), [2, 1, 0])

            elif case == "Case coronal: IS-RL-AP":
                mask = np.transpose(np.stack(segm), [1, 2, 0])
            else:
                print('ERROR')

            return mask

        axial_cases = {
            ('AP', 'RL', 'IS'): "Case axial: AP-RL-IS",
            ('IS', 'AP', 'RL'): "Case sagittal: IS-AP-RL",
            ('IS', 'RL', 'AP'): "Case coronal: IS-RL-AP"
        }

        mapped_orientation = self.map_orientation(orientation)
        # print(f"mapped_orientation {mapped_orientation}")

        if mapped_orientation in axial_cases:
            # print(f"{axial_cases[mapped_orientation]}")
            case = axial_cases[mapped_orientation]
            final_mask = transpose_mask(case, segm)
        else:
            print("Orientation not recognized.")
            final_mask = segm

        return final_mask

    def _is_current_model_3D(self):
        if self.classifications[int(self.curImage)] == 'None':
            return False
        dimensionality = get_model_detail(self.model_details, self.classifications[int(self.curImage)], 'dimensionality', None)
        if dimensionality is None:
            model, _ = self.get_model_for_class(self.classifications[int(self.curImage)], True, True)
            self.setSplash(False, 0, 1, '') # clear splash
            dimensionality = model.data_dimensionality
        return str(dimensionality) == '3'

    def _get_IL_contrast_volumes_3D(self):
        """Return an OrderedDict mapping 'image', 'image2', ... to the contrast volumes
        to use for incremental learning, in the same order used by getSegmentedMasks_3D()
        when calling the segmenter."""
        volumes = OrderedDict()
        volumes['image'] = self.additional_contrasts.get(self.current_contrast, self.medical_volume)
        contrast_index = 2
        for contrast_name, contrast_volume in self.additional_contrasts.items():
            if contrast_name == self.current_contrast:
                continue
            volumes[f'image{contrast_index}'] = contrast_volume
            contrast_index += 1
        return volumes

    def _get_IL_contrast_slices_2D(self, image_index):
        """Return an OrderedDict mapping 'image', 'image2', ... to the contrast slices
        to use for incremental learning, in the same order used by getSegmentedMasks()
        when calling the segmenter."""
        slices = OrderedDict()
        slices['image'] = self.imList[image_index]
        contrast_index = 2
        for contrast_name, contrast_volume in self.additional_contrasts.items():
            if contrast_name == self.current_contrast:
                continue
            slices[f'image{contrast_index}'] = contrast_volume.volume[:, :, image_index]
            contrast_index += 1
        return slices

    def calcOutputData(self, setSplash=False):
        imSize = self.image.shape

        allMasks = {}
        diceScores = []
        n_voxels = []

        dataForTraining = {}
        segForTraining = {}

        roi_names = self.roiManager.get_roi_names()
        current_roi_index = 0

        slices_with_rois = set()

        originalSegmentationMasks = {}

        for roiName in self.roiManager.get_roi_names():
            print("roiName: ", roiName)

            if setSplash:
                self.setSplash(True, current_roi_index, len(roi_names), "Calculating maps...")
                current_roi_index += 1

            if self._is_current_model_3D():

                contrast_volumes = self._get_IL_contrast_volumes_3D()
                masklist = np.zeros(self.medical_volume.shape, dtype=np.uint8)

                # print("len imList: ", len(self.imList))

                image_indices_3D = range(0, len(self.imList))

                # print('imageIndex: ', imageIndex)
                # print('self.medical_volume.shape[:2]: ', self.medical_volume.shape[:2])

                for index in range(len(self.imList)):
                    if self.roiManager.contains(roiName, index):
                        roi = self.roiManager.get_mask(roiName, index)
                        masklist[:, :, index]=roi
                        slices_with_rois.add(index)
                       
                count = 0
                for index in range(len(self.imList)):
                    if self.classifications[index] != 'None':
                        count = 0
                    else:
                        count += 1

                if image_indices_3D not in originalSegmentationMasks and count == 0:
                    originalSegmentationMasks = self.getSegmentedMasks_3D(image_indices_3D, False, True)

                try:
                    # print("try")
                    originalSegmentation = originalSegmentationMasks[roiName][:,:,image_indices_3D]
                except KeyError:
                    originalSegmentation = None

                if originalSegmentation is not None:
                    # diceScores.append(calc_dice_score(originalSegmentation, masklist))
                    # print('originalSegmentationMasks shape: ', originalSegmentationMasks[roiName].shape)
                    # print('originalSegmentation shape: ', originalSegmentation.shape)
                    # print('masklist shape: ', masklist.shape)

                    diceScores.append(calc_dice_score(originalSegmentation, masklist))
                    n_voxels.append(np.sum(masklist[:]))

                # TODO: maybe add this to the training according to the dice score?
                # count=0
                for index in range(len(self.imList)):
                    classification_name = self.classifications[index]

                    if classification_name not in dataForTraining:
                        dataForTraining[classification_name] = {}
                        segForTraining[classification_name] = {}
                    
                    if index not in dataForTraining[classification_name]:
                        dataForTraining[classification_name][index] = {
                            contrast_key: contrast_volume.volume[:, :, index]
                            for contrast_key, contrast_volume in contrast_volumes.items()
                        }
                        segForTraining[classification_name][index] = {roiName: masklist[:,:,index]}

                orientation=nib.aff2axcodes(self.affine)

                npMask = self.process_orientation(orientation, masklist)

                allMasks[roiName] = npMask
                torch.cuda.empty_cache()

            else:
                masklist = []
                for imageIndex in range(len(self.imList)):
                    roi = np.zeros(imSize, dtype=np.uint8)
                    if self.roiManager.contains(roiName, imageIndex):
                        roi = self.roiManager.get_mask(roiName, imageIndex)

                    if roi.any():
                        slices_with_rois.add(imageIndex) # add the slice to the set if any voxel is nonzero
                        if imageIndex not in originalSegmentationMasks and self.classifications[imageIndex] != 'None':
                            #print(imageIndex)
                            originalSegmentationMasks[imageIndex] = self.getSegmentedMasks(imageIndex, False, True)

                    masklist.append(roi)
                    try:
                        originalSegmentation = originalSegmentationMasks[imageIndex][roiName]
                    except KeyError:
                        originalSegmentation = None

                    if originalSegmentation is not None:
                        diceScores.append(calc_dice_score(originalSegmentation, roi))
                        n_voxels.append(np.sum(roi))
                        #print(diceScores)

                    # TODO: maybe add this to the training according to the dice score?
                    classification_name = self.classifications[imageIndex]
                    if classification_name not in dataForTraining:
                        dataForTraining[classification_name] = {}
                        segForTraining[classification_name] = {}
                    if imageIndex not in dataForTraining[classification_name]:
                        dataForTraining[classification_name][imageIndex] = self._get_IL_contrast_slices_2D(imageIndex)
                        segForTraining[classification_name][imageIndex] = {}

                    segForTraining[classification_name][imageIndex][roiName] = roi
                
                npMask = np.transpose(np.stack(masklist), [1, 2, 0])
                allMasks[roiName] = npMask

        # cleanup empty slices and slices that were already used for training
        for classification_name in dataForTraining:
            # print('Slices available for', classification_name, ':', list(dataForTraining[classification_name].keys()))
            for imageIndex in list(dataForTraining[classification_name]): # get a list of keys to be able to delete from dict
                if imageIndex not in slices_with_rois or imageIndex in self.slicesUsedForTraining:
                    del dataForTraining[classification_name][imageIndex]
                    del segForTraining[classification_name][imageIndex]
            # print('Slices after cleanup', list(dataForTraining[classification_name].keys()))

        if self._is_current_model_3D():
            for key in dataForTraining:
                indices = sorted(dataForTraining[key].keys())
                contrast_keys = dataForTraining[key][indices[0]].keys()
                dataForTraining[key] = {
                    contrast_key: np.stack([dataForTraining[key][i][contrast_key] for i in indices], axis=0)
                    for contrast_key in contrast_keys
                }

            for key in segForTraining:

                for sub_key in segForTraining[key][0].keys(): 
                    images = [segForTraining[key][i][sub_key] for i in sorted(segForTraining[key].keys())]
                    stacked = np.stack(images, axis=0)
                    segForTraining[key] = {sub_key: stacked}

            torch.cuda.empty_cache()

        diceScores = np.array(diceScores)
        n_voxels =np.array(n_voxels)
        if np.sum(n_voxels) == 0:
            average_dice = -1.0
        else:
            average_dice = np.average(diceScores) #, weights=n_voxels)
        print("Average Dice score", average_dice)

        return allMasks, dataForTraining, segForTraining, average_dice

    @pyqtSlot(str, str, bool)
    @snapshotSaver
    def copyRoi(self, originalName, newName, makeCopy=True):
        if makeCopy:
            self.roiManager.copy_roi(originalName, newName)
        else:
            self.roiManager.rename_roi(originalName, newName)
        self.updateRoiList()

    def _getCombineFunction(self, operator):
        if operator == 'Union':
            combine_fn = np.logical_or
        elif operator == 'Subtraction':
            combine_fn = lambda x,y: np.logical_and(x, np.logical_not(y))
        elif operator == 'Intersection':
            combine_fn = np.logical_and
        elif operator == 'Exclusion':
            combine_fn = np.logical_xor
        return combine_fn

    @pyqtSlot(str, str, str, str)
    @snapshotSaver
    def combineRoi(self, roi1, roi2, operator, dest_roi):
        self.combineMultiRoi([roi1, roi2], operator, dest_roi)

    @pyqtSlot(list, str, str)
    @snapshotSaver
    def combineMultiRoi(self, roi_list, operator, dest_roi):
        combine_fn = self._getCombineFunction(operator)
        if len(roi_list) < 2:
            return
        self.roiManager.generic_roi_combine(roi_list[0], roi_list[1], combine_fn, dest_roi)
        for i in range(2, len(roi_list)):
            self.roiManager.generic_roi_combine(dest_roi, roi_list[i], combine_fn, dest_roi)
        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.updateRoiList()

    @pyqtSlot()
    @snapshotSaver
    def roiRemoveOverlap(self):
        curRoiName = self.getCurrentROIName()
        currentMask = self.getCurrentMask()
        currentNotMask = np.logical_not(currentMask)
        for key_tuple, mask in self.roiManager.all_masks(image_number=self.curImage):
            if key_tuple[0] == curRoiName: continue
            self.roiManager.set_mask(key_tuple[0], key_tuple[1], np.logical_and(mask, currentNotMask))

        self.updateMasksFromROIs()
        self.reblit()

    @pyqtSlot()
    def delete_current_subregion(self):
        self.setCurrentSubregion(None)
        self.reblit()

    @pyqtSlot()
    def delete_all_subregions(self):
        if self.roiManager is None:
            return
        self.roiManager.clear_all_autosegment_subregions()
        self.removeSubregion()
        self.redraw()

    @pyqtSlot()
    def copy_all_subregions(self):
        subregion = self.getCurrentSubregion()
        if subregion is None:
            return

        for image_index in range(len(self.imList)):
            self.roiManager.set_autosegment_subregion(image_index, subregion)


    #########################################################################################
    ###
    ### ROI modifications
    ###
    #########################################################################################

    @snapshotSaver
    def simplify(self):
        r = self.getCurrentROI()
        self.setCurrentROI(r.getSimplifiedSpline())
        self.redraw() # this also updates the contour painters

    @snapshotSaver
    def optimize(self):
        r = self.getCurrentROI()
        center = r.getCenterOfMass()
        if center is None:
            print("No roi to optimize!")
            return

        newKnots = []
        for index, knot in enumerate(r.knots):
            # newKnot = self.optimizeKnot(center, knot)
            # newKnot = self.optimizeKnot2(knot, r.getKnot(index-1), r.getKnot(index+1))
            newKnot = self.optimizeKnot3(r, index)
            # newKnot = self.optimizeKnotDL(knot)
            newKnots.append(newKnot)

        for index, knot in enumerate(r.knots):
            r.replaceKnot(index, newKnots[index])
        self.reblit()

    # optimizes a knot along an (approximatE) normal to the curve
    def optimizeKnot2(self, knot, prevKnot, nextKnot):

        print("optimizeKnot2")

        optim_region = 5
        optim_region_points = optim_region * 4  # subpixel resolution

        # special case vertical line
        if prevKnot[0] == nextKnot[0]:
            # optimize along a horizontal line
            ypoints = knot[1] * np.ones((2 * optim_region_points))

            # define inside/outside
            if knot[0] < prevKnot[0]:
                xpoints = np.linspace(knot[0] + optim_region, knot[0] - optim_region, 2 * optim_region_points)
            else:
                xpoints = np.linspace(knot[0] - optim_region, knot[0] + optim_region, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        elif prevKnot[1] == nextKnot[1]:  # special case horizontal line
            # optimize along a horizontal line
            xpoints = knot[0] * np.ones((2 * optim_region_points))
            if knot[1] < prevKnot[1]:
                ypoints = np.linspace(knot[1] + optim_region, knot[1] - optim_region, 2 * optim_region_points)
            else:
                ypoints = np.linspace(knot[1] - optim_region, knot[1] + optim_region, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        else:
            slope = (nextKnot[1] - prevKnot[1]) / (nextKnot[0] - prevKnot[0])
            slope_perpendicular = -1 / slope
            x_dist = np.sqrt(optim_region / (
                    slope_perpendicular ** 2 + 1))  # solving the system (y1-y0) = m(x1-x0) and (y1-y0)^2 + (x1-x0)^2 = d

            # define inside*outside perimeter. Check line intersection. Is this happening on the right or on the left of the point? Right: go from high x to low x
            # x_intersection = (slope_perpendicular*knot[0] - knot[1] - slope*prevKnot[0] + prevKnot[1])/(slope_perpendicular-slope)
            # print knot[0]
            # print x_intersection
            # if x_intersection > knot[0]: x_dist = -x_dist

            x_min = knot[0] - x_dist
            x_max = knot[0] + x_dist
            y_min = knot[1] - slope_perpendicular * x_dist
            y_max = knot[1] + slope_perpendicular * x_dist
            xpoints = np.linspace(x_min, x_max, 2 * optim_region_points)
            ypoints = np.linspace(y_min, y_max, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        diffz = np.diff(z) / (np.abs(np.linspace(-optim_region, +optim_region, len(z) - 1)) + 1) ** (1 / 2)

        #            f = plt.figure()
        #            plt.subplot(121)
        #            plt.plot(z)
        #            plt.subplot(122)
        #            plt.plot(diffz)

        # find sharpest bright-to-dark transition. Maybe check if there are similar transitions in the line and only take the closest one
        minDeriv = np.argmax(np.abs(diffz)) + 1
        return (xpoints[minDeriv], ypoints[minDeriv])

    # optimizes a knot along an (approximate) normal to the curve, going from inside the ROI to outside
    def optimizeKnot3(self, roi, knotIndex):

        knot = roi.getKnot(knotIndex)
        nextKnot = roi.getKnot(knotIndex + 1)
        prevKnot = roi.getKnot(knotIndex - 1)

        # print "optimizeKnot3"

        optim_region = 5
        optim_region_points = optim_region * 4  # subpixel resolution

        # special case vertical line
        if prevKnot[0] == nextKnot[0]:
            # optimize along a horizontal line
            ypoints = knot[1] * np.ones((2 * optim_region_points))

            # define inside/outside
            if knot[0] < prevKnot[0]:
                xpoints = np.linspace(knot[0] + optim_region, knot[0] - optim_region, 2 * optim_region_points)
            else:
                xpoints = np.linspace(knot[0] - optim_region, knot[0] + optim_region, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        elif prevKnot[1] == nextKnot[1]:  # special case horizontal line
            # optimize along a horizontal line
            xpoints = knot[0] * np.ones((2 * optim_region_points))
            if knot[1] < prevKnot[1]:
                ypoints = np.linspace(knot[1] + optim_region, knot[1] - optim_region, 2 * optim_region_points)
            else:
                ypoints = np.linspace(knot[1] - optim_region, knot[1] + optim_region, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        else:
            slope = (nextKnot[1] - prevKnot[1]) / (nextKnot[0] - prevKnot[0])
            slope_perpendicular = -1 / slope
            x_dist = np.sqrt(optim_region / (
                    slope_perpendicular ** 2 + 1))  # solving the system (y1-y0) = m(x1-x0) and (y1-y0)^2 + (x1-x0)^2 = d

            # this point is just on the right of our knot.
            test_point_x = knot[0] + 1
            test_point_y = knot[1] + slope_perpendicular * 1

            # if the point is inside the ROI, then calculate the line from right to left
            if roi.isPointInside((test_point_x, test_point_y)):
                x_dist = -x_dist

            # define inside*outside perimeter. Check line intersection. Is this happening on the right or on the left of the point? Right: go from high x to low x
            # x_intersection = (slope_perpendicular*knot[0] - knot[1] - slope*prevKnot[0] + prevKnot[1])/(slope_perpendicular-slope)
            # print knot[0]
            # print x_intersection
            # if x_intersection > knot[0]: x_dist = -x_dist

            x_min = knot[0] - x_dist
            x_max = knot[0] + x_dist
            y_min = knot[1] - slope_perpendicular * x_dist
            y_max = knot[1] + slope_perpendicular * x_dist
            xpoints = np.linspace(x_min, x_max, 2 * optim_region_points)
            ypoints = np.linspace(y_min, y_max, 2 * optim_region_points)
            z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)

        # sensitive to bright-to-dark
        # diffz = np.diff(z) / (np.abs(np.linspace(-optim_region,+optim_region,len(z)-1))+1)**(1/2)

        # sensitive to all edges
        diffz = -np.abs(np.diff(z)) / (np.abs(np.linspace(-optim_region, +optim_region, len(z) - 1)) + 1) ** (1 / 2)

        #        f = plt.figure()
        #        plt.subplot(121)
        #        plt.plot(z)
        #        plt.subplot(122)
        #        plt.plot(diffz)

        # find sharpest bright-to-dark transition. Maybe check if there are similar transitions in the line and only take the closest one
        minDeriv = np.argmin(diffz)
        # print minDeriv
        return (xpoints[minDeriv], ypoints[minDeriv])

    # optimizes a knot along a radius from the center of the ROI
    def optimizeKnot(self, center, knot):

        optim_region = 5  # voxels

        distanceX = knot[0] - center[0]
        distanceY = knot[1] - center[1]
        npoints = int(np.max([abs(2 * distanceX), abs(2 * distanceY)]))
        xpoints = center[0] + np.linspace(0, 2 * distanceX, npoints)
        ypoints = center[1] + np.linspace(0, 2 * distanceY, npoints)

        # restrict to region aroung the knot
        minIndex = np.max([0, int(npoints / 2 - optim_region)])
        maxIndex = np.min([int(npoints / 2 + optim_region), npoints])

        xpoints = xpoints[minIndex:maxIndex]
        ypoints = ypoints[minIndex:maxIndex]

        # print xpoints
        # print ypoints
        z = ndimage.map_coordinates(self.image, np.vstack((ypoints, xpoints))).astype(np.float32)
        diffz = np.diff(z) / (np.abs(np.array(range(len(z) - 1)) - (len(z) - 1) / 2) ** 2 + 1)

        #        f = plt.figure()
        #        plt.subplot(121)
        #        plt.plot(z)
        #        plt.subplot(122)
        #        plt.plot(diffz)

        # find sharpest bright-to-dark transition. Maybe check if there are similar transitions in the line and only take the closest one
        minDeriv = np.argmin(diffz) + 1
        return (xpoints[minDeriv], ypoints[minDeriv])

    # No @snapshotSaver: snapshot is saved in the calling function
    def addPoint(self, spline, event):
        self.currentPoint = spline.addKnot((event.xdata, event.ydata))
        self.reblit()

    # No @snapshotSaver: snapshot is saved in the calling function
    def movePoint(self, spline, event):
        if self.currentPoint is None:
            return
        spline.replaceKnot(self.currentPoint, (event.xdata, event.ydata))
        self.reblit()

    @pyqtSlot()
    @snapshotSaver
    def clearCurrentROI(self):
        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            roi = self.getCurrentROI()
            roi.removeAllKnots()
        elif self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.roiManager.clear_mask(self.getCurrentROIName(), self.curImage)
            self.activeMask = None
        self.reblit()

    @snapshotSaver
    def _currentMaskOperation(self, operation_function):
        """
        Applies a generic operation to the current mask. operation_function is a function that accepts the mask as parameter
        and returns the new mask
        """
        if not self.editMode == ToolboxWindow.EDITMODE_MASK: return
        currentMask = self.getCurrentMask()
        newMask = operation_function(currentMask)
        self.setCurrentMask(newMask)
        if self.activeMask is None:
            self.updateMasksFromROIs()
        else: # only update the active mask
            self.activeMask = newMask.copy()
        self.reblit()

    @pyqtSlot()
    def maskGrow(self):
        self._currentMaskOperation(binary_dilation)

    @pyqtSlot()
    def maskShrink(self):
        self._currentMaskOperation(binary_erosion)

    @pyqtSlot(int)
    def maskDespeckle(self, radius):
        #self._currentMaskOperation(lambda mask: area_opening(mask, radius**2))
        self._currentMaskOperation(functools.partial(area_opening, area_threshold=radius ** 2))

    @pyqtSlot(int)
    def maskFillHoles(self, radius):
        #self._currentMaskOperation(lambda mask: area_closing(mask, radius ** 2))
        self._currentMaskOperation(functools.partial(area_closing, area_threshold=radius ** 2))

    @pyqtSlot()
    @snapshotSaver
    @separate_thread_decorator
    def samAutoRefine(self):
        if not self.editMode == ToolboxWindow.EDITMODE_MASK or \
                not self.roiManager:
            return

        def progress_callback(current, maximum):
            self.setSplash(True, current, maximum, "SAM autorefine")

        try:
            new_mask = sam_api.SAM_refine(self.image, self.getCurrentMask(), self.get_sam(), GlobalConfig['SAM_PROMPT_MODE'].lower(), progress_callback)
        except Exception as e:
            print("Error in SAM autorefine:", e)
            self.alert("Error in SAM autorefine: " + str(e))
            self.setSplash(False)
            return

        self.setCurrentMask(new_mask)
        self.updateMasksFromROIs()
        self.reblit()
        self.setSplash(False)

    def samPropagateBlock(self, all_rois, inplace=True):
        """ Propagate existing masks across each ROI's own known extent (the range between
        its farthest segmented slice above and its farthest segmented slice below the
        current slice) with SAM2 (dafne_sam2.public_api.SAM_propagate), using every
        already-segmented slice of that ROI as anchors/support.

        all_rois=False: only the currently selected ROI is propagated (as before).
        all_rois=True: every ROI in the roiManager is handed to a single SAM_propagate call
        -- this is the "act on all ROIs simultaneously" mode, using SAM_propagate's own
        per-ROI dict support rather than looping calls. A ROI with no existing masks, or with
        masks only above or only below the current slice, is skipped rather than aborting the
        whole operation; the alert only fires if no ROI qualifies at all.

        Always prompts SAM2 with the anchor masks themselves (prompt_kind='mask'), trusted
        as-is (refine_mask_prompt=False) -- these are the user's own confirmed/edited
        masks, not pseudo-labels to second-guess, so this ignores GlobalConfig['SAM_PROMPT_MODE'].

        inplace=True: overwrite each propagated ROI's masks, in that ROI's own range, in the
        roiManager.
        inplace=False: leave the roiManager untouched, and return
        dict[roi_name -> volume (H, W, len(imList))], one entry per ROI that had a result,
        each volume holding the propagated masks in that ROI's range and zeros elsewhere. """
        if not self.roiManager:
            return None

        if all_rois:
            roi_names = self.roiManager.get_roi_names()
        else:
            current_roi_name = self.getCurrentROIName()
            roi_names = [current_roi_name] if current_roi_name else []
        if not roi_names:
            return None

        masks_by_roi = {}
        z_bounds = {}
        for roi_name in roi_names:
            anchors = {int(image_number): mask for (name, image_number), mask
                      in self.roiManager.all_masks(roi_name=roi_name) if np.any(mask)}
            if not anchors:
                continue
            masks_above, masks_above_index, masks_below, masks_below_index = self._get_masks_above_below(roi_name)
            if not masks_above or not masks_below:
                continue
            masks_by_roi[roi_name] = anchors
            z_bounds[roi_name] = (masks_above_index[-1], masks_below_index[-1])

        if not masks_by_roi:
            self.alert('No ROI has both a segmented slice above and a segmented slice below the current one to use as support for SAM propagation')
            return None

        def progress_callback(current, maximum):
            self.setSplash(True, current, maximum, "SAM propagate")

        try:
            result = sam_api.SAM_propagate(
                np.stack(self.imList), masks_by_roi, self.get_sam(),
                z_bounds=z_bounds,
                prompt_kind='mask', refine_mask_prompt=False,
                progress_callback=progress_callback)
        except Exception as e:
            print("Error in SAM propagate:", e)
            self.alert("Error in SAM propagate: " + str(e))
            self.setSplash(False)
            return None

        self.setSplash(False)

        if inplace:
            for roi_name, propagated in result.items():
                lo, hi = z_bounds[roi_name]
                for image_number in range(lo, hi + 1):
                    mask = propagated.get(image_number)
                    if mask is None:
                        mask = np.zeros(self.image.shape, dtype=np.uint8)
                    self.roiManager.set_mask(roi_name, image_number, mask.astype(np.uint8))
            self.updateMasksFromROIs()
            self.reblit()
            return None

        out_volumes = {}
        for roi_name, propagated in result.items():
            out_volume = np.zeros((self.image.shape[0], self.image.shape[1], len(self.imList)), dtype=np.uint8)
            for image_number, mask in propagated.items():
                out_volume[:, :, image_number] = mask
            out_volumes[roi_name] = out_volume
        return out_volumes

    @pyqtSlot(bool)
    @snapshotSaver
    @separate_thread_decorator
    def maskAutoThreshold(self, apply_to_all=False):
        if not self.editMode == ToolboxWindow.EDITMODE_MASK or \
                not self.roiManager:
            return
        self.setSplash(True, 0, 2, "Calculating threshold mask...")
        # Calculate the mask after bias correction
        bias_corrected_image = biascorrection_image(self.image)
        threshold_mask = sitk.GetArrayFromImage(
                            sitk.OtsuThreshold(
                            sitk.GetImageFromArray(bias_corrected_image), 0, 1, 200))

        self.setSplash(True, 1, 2, "Applying threshold...")
        if apply_to_all:
            for mask_key_tuple, mask in self.roiManager.all_masks(image_number=self.curImage):
                roi_name = mask_key_tuple[0]
                print("Processing", roi_name)
                new_mask = np.logical_and(mask, threshold_mask)
                self.roiManager.set_mask(roi_name, self.curImage, new_mask)
        else:
            mask = self.getCurrentMask()
            if mask is None:
                self.setSplash(False)
                return
            new_mask = np.logical_and(mask, threshold_mask)
            self.setCurrentMask(new_mask)

        self.setSplash(True, 2, 2, "Done!")
        self.updateMasksFromROIs()
        self.reblit()
        self.setSplash(False)




    #####################################################################################################
    ###
    ### Elastix
    ###
    #####################################################################################################

    @separate_thread_decorator
    def calcTransforms(self):
        if not self.registrationManager: return
        def local_setSplash(image_number):
            self.setSplash(True, image_number, len(self.imList), 'Registering images...')

        local_setSplash(0)
        self.registrationManager.calc_transforms(local_setSplash)
        self.setSplash(False, 0, len(self.imList), 'Registering images...')


    def propagateAll(self):
        while self.curImage < len(self.imList) - 1:
            self.propagate()
            plt.pause(.000001)

    def propagateBackAll(self):
        while self.curImage > 0:
            self.propagateBack()
            plt.pause(.000001)

    @snapshotSaver
    #@separate_thread_decorator
    def propagate(self):
        if self.curImage >= len(self.imList) - 1: return
        if not self.registrationManager: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]

        self.setSplash(True, 0, 3)


        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            curROI = self.getCurrentROI()
            if curROI is None:
                self.setSplash(False, 0, 0)
                return
            nextROI = self.getCurrentROI(+1)
            knotsOut = self.registrationManager.run_transformix_knots(curROI.knots,
                                                                      self.registrationManager.get_transform(int(self.curImage)))

            if len(nextROI.knots) < 3:
                nextROI.removeAllKnots()
                nextROI.addKnots(knotsOut)
            else:
                for k in knotsOut:
                    i = nextROI.findNearestKnot(k)
                    oldK = nextROI.getKnot(i)
                    newK = ((oldK[0] + k[0]) / 2, (oldK[1] + k[1]) / 2)
                    # print "oldK", oldK, "new", k, "mid", newK
                    nextROI.replaceKnot(i, newK)
        elif self.editMode == ToolboxWindow.EDITMODE_MASK:
            mask_in = self.getCurrentMask()
            if mask_in is None:
                self.setSplash(False, 0, 0)
                return
            # Note: we are using the inverse transform, because the transforms are originally calculated to
            # transform points, which is the inverse as transforming images
            mask_out = self.registrationManager.run_transformix_mask(mask_in,
                                                                     self.registrationManager.get_inverse_transform(int(self.curImage+1)))
            self.setCurrentMask(mask_out, +1)


        self.curImage += 1
        self.displayImage(int(self.curImage), self.cmap, redraw=False)
        self.setSplash(True, 1, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            self.simplify()
            self.setSplash(True, 2, 3)
            self.optimize()

        self.redraw()

        self.setSplash(False, 3, 3)

    @snapshotSaver
    #@separate_thread_decorator
    def propagateBack(self):
        if self.curImage < 1: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]

        self.setSplash(True, 0, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            curROI = self.getCurrentROI()
            if curROI is None:
                self.setSplash(False, 0, 0)
                return
            nextROI = self.getCurrentROI(-1)
            knotsOut = self.registrationManager.run_transformix_knots(curROI.knots,
                                                                      self.registrationManager.get_inverse_transform(int(self.curImage)))

            if len(nextROI.knots) < 3:
                nextROI.removeAllKnots()
                nextROI.addKnots(knotsOut)
            else:
                for k in knotsOut:
                    i = nextROI.findNearestKnot(k)
                    oldK = nextROI.getKnot(i)
                    newK = ((oldK[0] + k[0]) / 2, (oldK[1] + k[1]) / 2)
                    nextROI.replaceKnot(i, newK)
        elif self.editMode == ToolboxWindow.EDITMODE_MASK:
            mask_in = self.getCurrentMask()
            if mask_in is None:
                self.setSplash(False, 0, 0)
                return
            # Note: we are using the inverse transform, because the transforms are originally calculated to
            # transform points, which is the inverse as transforming images
            mask_out = self.registrationManager.run_transformix_mask(mask_in,
                                                                     self.registrationManager.get_transform(int(self.curImage-1)))
            self.setCurrentMask(mask_out, -1)

        self.setSplash(True, 1, 3)

        self.curImage -= 1
        self.displayImage(int(self.curImage), self.cmap, redraw=False)

        self.setSplash(True, 2, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            self.simplify()
            self.setSplash(True, 3, 3)
            self.optimize()

        self.redraw()

        self.setSplash(False, 3, 3)

    def _get_masks_above_below(self, roi_name=None):
        """ roi_name=None (default): use the currently selected ROI (as getCurrentMask does).
        Pass an explicit roi_name to inspect a different ROI without touching the toolbox
        selection -- needed for all_rois propagation, which runs from a background thread. """
        roi_name = roi_name if roi_name is not None else self.getCurrentROIName()
        if not self.roiManager or not roi_name:
            return [], [], [], []
        self.curImage = int(self.curImage)
        masks_above = []
        masks_above_index = []
        for i in range(self.curImage - 1, -1, -1):
            m = self.roiManager.get_mask(roi_name, i)
            if np.any(m):
                masks_above.append(m)
                masks_above_index.append(i)

        masks_below = []
        masks_below_index = []
        for i in range(self.curImage + 1, len(self.imList)):
            m = self.roiManager.get_mask(roi_name, i)
            if np.any(m):
                masks_below.append(m)
                masks_below_index.append(i)

        return masks_above, masks_above_index, masks_below, masks_below_index

    def _calculateInterpolatedMask(self, roi_name=None):
        # find ROIs above and below the current image

        masks_above, masks_above_index, masks_below, masks_below_index = self._get_masks_above_below(roi_name)

        if not masks_above and not masks_below:
            print('No masks to interpolate')
            return np.zeros(self.image.shape, dtype=np.uint8)
        if not masks_above or not masks_below:
            # all the masks are either above or below the current image: don't interpolate as the results are bad
            #print('Nearest neighbor')
            return (masks_above + masks_below)[0]
        else: # len(masks_above) < 2 or len(masks_below) < 2: Let's disable cubic interpolation
            #print('Linear interpolation')
            # We have fewer than 2 masks above and below. Can't use cubic interpolation. Just do linear
            # interpolation between the closest masks

            spline_list_1 = mask_to_trivial_splines(masks_above[0], spacing=4)
            spline_list_2 = mask_to_trivial_splines(masks_below[0], spacing=4)
            #print('Number of splines', len(spline_list_1))
            index1 = masks_above_index[0]
            index2 = masks_below_index[0]
            if len(spline_list_1) != len(spline_list_2):
                self.alert('Different number of subrois in neighboring regions')
                return np.zeros(self.image.shape, dtype=np.uint8)

            splines_list = masks_splines_to_splines_masks([spline_list_1, spline_list_2])
            out_mask = np.zeros(self.image.shape, dtype=np.uint8)
            for subroi_spline in splines_list:
                out_spline = SplineInterpROIClass()
                spline1 = subroi_spline[0]
                spline2 = subroi_spline[1]

                current_index = self.curImage
                for knot1, knot2 in zip(spline1.knots, spline2.knots):
                    f_x = interp1d([index1, index2], [knot1[0], knot2[0]], kind='linear')
                    f_y = interp1d([index1, index2], [knot1[1], knot2[1]], kind='linear')
                    out_spline.addKnot((f_x(current_index), f_y(current_index)))
                out_mask += out_spline.toMask(self.image.shape)
                out_mask = (out_mask > 0).astype(np.uint8)
                out_mask = binary_dilation(out_mask)
            return out_mask
        if 0: # cubic interpolation disabled
            # we have at least 2 slices above and 2 slices below: cubic interpolation
            #print('Cubic interpolation')
            spline_list_1 = mask_to_trivial_splines(masks_above[1], spacing=4)
            spline_list_2 = mask_to_trivial_splines(masks_above[0], spacing=4)
            spline_list_3 = mask_to_trivial_splines(masks_below[0], spacing=4)
            spline_list_4 = mask_to_trivial_splines(masks_below[1], spacing=4)
            # print('Number of splines', len(spline_list_1))
            index1 = masks_above_index[1]
            index2 = masks_above_index[0]
            index3 = masks_below_index[0]
            index4 = masks_below_index[1]
            if any([len(spline_list_1) != len(spline_list_2), len(spline_list_1) != len(spline_list_3), len(spline_list_1) != len(spline_list_4)]):
                self.alert('Different number of subrois in neighboring regions')
                return np.zeros(self.image.shape, dtype=np.uint8)

            splines_list = masks_splines_to_splines_masks([spline_list_1, spline_list_2, spline_list_3, spline_list_4])
            out_mask = np.zeros(self.image.shape, dtype=np.uint8)
            for subroi_spline in splines_list:
                out_spline = SplineInterpROIClass()
                spline1 = subroi_spline[0]
                spline2 = subroi_spline[1]
                spline3 = subroi_spline[2]
                spline4 = subroi_spline[3]

                current_index = self.curImage
                for knot1, knot2, knot3, knot4 in zip(spline1.knots, spline2.knots, spline3.knots, spline4.knots):
                    f_x = interp1d([index1, index2, index3, index4], [knot1[0], knot2[0], knot3[0], knot4[0]], kind='cubic')
                    f_y = interp1d([index1, index2, index3, index4], [knot1[1], knot2[1], knot3[1], knot4[1]], kind='cubic')
                    out_spline.addKnot((f_x(current_index), f_y(current_index)))

                out_mask += out_spline.toMask(self.image.shape)
                out_mask = (out_mask > 0).astype(np.uint8)
                out_mask = binary_dilation(out_mask)

            return out_mask


    def _registerMask(self, roi_name=None):
        if self.registrationManager is None:
            return np.zeros(self.image.shape, dtype=np.uint8)

        self.setSplash(True, 0, 1, 'Calculating registration')

        masks_above, masks_above_index, masks_below, masks_below_index = self._get_masks_above_below(roi_name)

        if not masks_above and not masks_below:
            return np.zeros(self.image.shape, dtype=np.uint8)

        mask_above = masks_above[0]
        mask_below = masks_below[0]

        mask_above_index = masks_above_index[0]
        mask_below_index = masks_below_index[0]

        registered_mask_above = None
        if mask_above is not None:
            registered_mask_above = mask_above
            for i in range(mask_above_index, self.curImage):
                # Note: we are using the inverse transform, because the transforms are originally calculated to
                # transform points, which is the inverse as transforming images
                registered_mask_above = self.registrationManager.run_transformix_mask(registered_mask_above,
                                                                    self.registrationManager.get_inverse_transform(i+1))

        registered_mask_below = None
        if mask_below is not None:
            registered_mask_below = mask_below
            for i in range(mask_below_index, self.curImage, -1):
                # Note: we are using the inverse transform, because the transforms are originally calculated to
                # transform points, which is the inverse as transforming images
                registered_mask_below = self.registrationManager.run_transformix_mask(registered_mask_below,
                                                                         self.registrationManager.get_transform(
                                                                             i - 1))

        self.setSplash(False)
        if registered_mask_above is None:
            return registered_mask_below
        elif registered_mask_below is None:
            return registered_mask_above
        else:
            return binary_dilation(mask_average([registered_mask_above, registered_mask_below],
                                [self.curImage-mask_below_index, mask_above_index-self.curImage]))

    @pyqtSlot(str, bool)
    @snapshotSaver
    @separate_thread_decorator
    def interpolate(self, interpolation_method, all_rois):
        self.do_interpolate(interpolation_method, all_rois)

    def do_interpolate(self, interpolation_method, all_rois, set_splash=True):
        #if self.editMode == ToolboxWindow.EDITMODE_CONTOUR: return
        if interpolation_method == ToolboxWindow.INTERPOLATE_MASK_SAM:
            out_volumes = self._interpolate_block(interpolation_method, all_rois, inplace=False)
            if not out_volumes:
                return
            for roi_name, out_volume in out_volumes.items():
                self.roiManager.set_mask(roi_name, int(self.curImage), out_volume[:, :, int(self.curImage)])
            self.redraw()
            return

        if not self.roiManager:
            return
        if all_rois:
            roi_names = self.roiManager.get_roi_names()
        else:
            current_roi_name = self.getCurrentROIName()
            roi_names = [current_roi_name] if current_roi_name else []

        for i, roi_name in enumerate(roi_names):
            if set_splash:
                self.setSplash(True, i, len(roi_names), 'Interpolating masks...')
            if interpolation_method == ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE:
                new_mask = self._calculateInterpolatedMask(roi_name)
            elif interpolation_method == ToolboxWindow.INTERPOLATE_MASK_REGISTER:
                new_mask = self._registerMask(roi_name)
            elif interpolation_method == ToolboxWindow.INTERPOLATE_MASK_BOTH:
                interpolated_mask = self._calculateInterpolatedMask(roi_name)
                registered_mask = self._registerMask(roi_name)
                new_mask = binary_dilation(mask_average([interpolated_mask, registered_mask]))
            else:
                continue
            if not np.any(new_mask):
                # nothing to interpolate/register this ROI from (no segmented slice on one
                # side, or on both) -- leave it untouched instead of blanking it. Matters
                # most for all_rois, where most ROIs won't have data around every slice.
                continue
            self.roiManager.set_mask(roi_name, int(self.curImage), new_mask.astype(np.uint8))
            if set_splash:
                self.setSplash(False)
        self.redraw()

    @pyqtSlot(str, bool)
    @snapshotSaver
    @separate_thread_decorator
    def interpolate_block(self, interpolation_method, all_rois, inplace=True):
        self._interpolate_block(interpolation_method, all_rois, inplace)

    def _interpolate_block(self, interpolation_method, all_rois, inplace=True):
        """ Shared logic behind the interpolate_block slot -- factored out so
        do_interpolate's single-slice SAM branch can call it synchronously (interpolate_block
        itself is @separate_thread_decorator/@snapshotSaver-wrapped and so cannot be called
        for its return value). """
        #if self.editMode == ToolboxWindow.EDITMODE_CONTOUR: return

        if interpolation_method == ToolboxWindow.INTERPOLATE_MASK_SAM:
            # use SAM for volume interpolation; samPropagateBlock derives each ROI's own
            # extent (and, with all_rois, skips any ROI that doesn't have both a segmented
            # slice above and below), so it doesn't need the current-ROI-only guard below.
            return self.samPropagateBlock(all_rois, inplace)

        # there needs to be at least one segmented slice above and one segmented slice below
        # of the currently selected ROI -- this defines the block's slice range; all_rois
        # then interpolates every ROI, one by one, on each slice within that range.
        masks_above, masks_above_index, masks_below, masks_below_index = self._get_masks_above_below()
        if not masks_above or not masks_below:
            self.alert('Block interpolation only works if there is at least one segmented slice above and one segmented slice below')
            return None

        initial_index = masks_above_index[0] + 1
        final_index = masks_below_index[0] - 1
        for i in range(initial_index, final_index+1):
            self.setSplash(True, i-initial_index, final_index - initial_index + 1, 'Interpolating masks...')
            self.curImage = i
            self.displayImage(self.curImage)
            self.redraw()
            self.do_interpolate(interpolation_method, all_rois, set_splash=False)
        self.setSplash(False)
        return None

    @pyqtSlot(np.ndarray, dict, list)
    @snapshotSaver
    @separate_thread_decorator
    def transfer_roi(self, support_volume, masks, resolution):
        """ transfer a ROI from a support volume to the current slice """
        current_image = self.imList[int(self.curImage)]

        def progress_callback(current, maximum):
            self.setSplash(True, current, maximum, 'Transferring ROI...')

        transferred_rois = sam_api.transfer_slice(current_image, support_volume, masks, self.get_sam(), progress_callback=progress_callback)
        self.masksToRois(transferred_rois, int(self.curImage))
        self.setSplash(False)


    ##############################################################################################################
    ###
    ### Displaying
    ###
    ###############################################################################################################

    def gotoImageDialog(self):
        accepted, output = GenericInputDialog.show_dialog("Go to image", [
            GenericInputDialog.IntSpinInput("Image number", self.curImage, 0, len(self.imList) - 1),
        ], self.fig.canvas)
        if accepted:
            self.displayImage(output[0], redraw=True)

    def removeMasks(self):
        """ Remove the masks from the plot """
        try:
            self.maskImPlot.remove()
        except:
            pass
        self.maskImPlot = None

        try:
            self.maskOtherImPlot.remove()
        except:
            pass
        self.maskOtherImPlot = None

        try:
            self.brush_patch.remove()
        except:
            pass
        self.brush_patch = None

        self.activeMask = None
        self.otherMask = None

    def removeContours(self):
        """ Remove all the contours from the plot """
        self.activeRoiPainter.clear_patches(self.axes)
        self.sameRoiPainter.clear_patches(self.axes)
        self.otherRoiPainter.clear_patches(self.axes)

    def removeSubregion(self):
        """ Remove the autosegment subregion from the plot """
        if not self.region_rectangle:
            return

        try:
            self.region_rectangle.set_visible(False)
        except:
            pass

        try:
            self.region_rectangle.remove()
        except:
            pass

        self.region_rectangle = None


    def updateMasksFromROIs(self):
        roi_name = self.getCurrentROIName()
        mask_size = self.image.shape
        self.otherMask = np.zeros(mask_size, dtype=np.uint8)
        self.activeMask = np.zeros(mask_size, dtype=np.uint8)
        current_other_mask_index = 2
        mask_error = False
        for key_tuple, mask in self.roiManager.all_masks(image_number=self.curImage):
            mask_name = key_tuple[0]
            if mask_name == roi_name:
                if mask is not None:
                    self.activeMask = mask.copy()
            else:
                if mask is not None:
                    layer_mask = (current_other_mask_index*mask).astype(np.uint8)
                    try:
                        self.otherMask += layer_mask
                    except TypeError:
                        print("TypeError in otherMask addition")
                        mask_error = True
                        # probably a thread issue that makes otherMask None; just skip this mask and continue
                    current_other_mask_index += 1
        if mask_error:
            self.otherMask = None
        self.emit_mask_slice_changed()

    def emit_mask_changed(self):
        if not self.toolbox_window.is_3D_viewer_visible(): return
        if self.roiManager is None:
            return
        roi_name = self.getCurrentROIName()
        if not roi_name:
            return
        spacing = [self.resolution[0], self.resolution[1], self.resolution[2]]
        mask_shape = (self.image.shape[0], self.image.shape[1], len(self.imList))
        full_mask = np.zeros(mask_shape, dtype=np.uint8)
        for key_tuple, mask in self.roiManager.all_masks(roi_name=roi_name):
            if mask is None: continue
            mask_slice = key_tuple[1]
            full_mask[:, :, mask_slice] = mask
        self.mask_changed.emit(spacing, full_mask)

        # labeled volume of the non-active ROIs; the label values (from 2 up) match the
        # coloring of the "other" masks in the main window
        other_mask = np.zeros(mask_shape, dtype=np.uint8)
        current_other_mask_index = 2
        for other_name in self.roiManager.get_roi_names():
            if other_name == roi_name:
                continue
            for key_tuple, mask in self.roiManager.all_masks(roi_name=other_name):
                if mask is None: continue
                other_mask[:, :, key_tuple[1]][mask > 0] = current_other_mask_index
            current_other_mask_index += 1
        self.other_mask_changed.emit(spacing, other_mask)

    def emit_viewer3d_data(self):
        """ Send the current state (displayed volume, slice position, masks) to the
            triplanar/3D viewer. Used when the viewer is shown and when the displayed
            volume changes (contrast or time frame switch). """
        if self.medical_volume is None:
            return
        display_volume = self.medical_volume
        if self.additional_contrasts:
            display_volume = self.additional_contrasts.get(self.current_contrast, self.medical_volume)
        self.volume_loaded_signal.emit([self.resolution[0], self.resolution[1], self.resolution[2]],
                                       display_volume.volume)
        self.displayed_slice_changed.emit(int(self.curImage))
        self.emit_mask_changed()

    @pyqtSlot(int)
    def viewer3d_slice_changed(self, slice_number):
        """ The user navigated to a new main-plane slice in the triplanar viewer. """
        if not self.imList or len(self.imList) == 0:
            return
        slice_number = int(max(0, min(slice_number, len(self.imList) - 1)))
        if int(self.curImage) == slice_number:
            return
        self.displayImage(slice_number, redraw=False)
        self.redraw()

    def emit_mask_slice_changed(self):
        if not self.toolbox_window.is_3D_viewer_visible(): return
        if self.roiManager is None:
            return

        slice_n = int(self.curImage)
        mask = self.getCurrentMask()
        if mask is None:
            return

        self.mask_slice_changed.emit(slice_n, mask)

    def drawMasks(self):
        """ Plot the masks for the current figure """
        # print("Draw masks", time.time())
        # frame = inspect.getouterframes(inspect.currentframe(), 2)
        # for info in frame:
        #     print("Trace", info[3])
        if self.activeMask is None or self.otherMask is None:
            self.updateMasksFromROIs()

        if self.activeMask is None or self.otherMask is None:
            return

        if not self.hideRois:  # if we hide the ROIs, clear all the masks
            active_mask = self.activeMask
            other_mask = self.otherMask
        else:
            active_mask = np.zeros_like(self.activeMask, dtype=np.uint8)
            other_mask = np.zeros_like(self.otherMask, dtype=np.uint8)

        if self.maskImPlot is None:
            original_xlim = self.axes.get_xlim()
            original_ylim = self.axes.get_ylim()
            self.maskImPlot = self.axes.imshow(active_mask, cmap=self.mask_layer_colormap,
                                               alpha=GlobalConfig['MASK_LAYER_ALPHA'],
                                               vmin=0, vmax=1, zorder=100, aspect=self.resolution[0]/self.resolution[1])
            try:
                self.axes.set_xlim(original_xlim)
                self.axes.set_ylim(original_ylim)
            except:
                pass
            self.maskImPlot.set_animated(True)

        self.maskImPlot.set_data(active_mask.astype(np.uint8))
        self.maskImPlot.set_alpha(GlobalConfig['MASK_LAYER_ALPHA'])
        self.axes.draw_artist(self.maskImPlot)

        if self.maskOtherImPlot is None:
            original_xlim = self.axes.get_xlim()
            original_ylim = self.axes.get_ylim()
            relativeAlphaROI = GlobalConfig['ROI_OTHER_COLOR'][3] / GlobalConfig['ROI_COLOR'][3]
            self.maskOtherImPlot = self.axes.imshow(other_mask,
                                                    alpha=relativeAlphaROI*GlobalConfig['MASK_LAYER_ALPHA'],
                                                    zorder=101, aspect=self.resolution[0]/self.resolution[1])
            try:
                self.axes.set_xlim(original_xlim)
                self.axes.set_ylim(original_ylim)
            except:
                pass
            self.maskOtherImPlot.set_animated(True)

        self.maskOtherImPlot.set_data(other_mask.astype(np.uint8))
        if GlobalConfig['USE_MULTIPLE_OTHER_COLORS']:
            other_colormap = hue_compass_colormap.generate_colormap(GlobalConfig['ROI_COLOR'], len(self.roiManager.allROIs)-1)
            self.maskOtherImPlot.set_cmap(other_colormap)
            self.maskOtherImPlot.set_clim(vmin=0, vmax=other_colormap.N - 1)
            #('Other mask max', other_mask.max())
        else:
            self.maskOtherImPlot.set_cmap(self.mask_layer_other_colormap)
            self.maskOtherImPlot.set_clim(vmin=0, vmax=1)
        self.maskOtherImPlot.set_alpha(GlobalConfig['MASK_LAYER_ALPHA'])
        self.axes.draw_artist(self.maskOtherImPlot)

    def updateContourPainters(self):
        # frame = inspect.getouterframes(inspect.currentframe(), 2)
        # for info in frame:
        #     print("Trace", info[3])


        self.activeRoiPainter.clear_rois(self.axes)
        self.otherRoiPainter.clear_rois(self.axes)
        self.sameRoiPainter.clear_rois(self.axes)
        if not self.roiManager or self.editMode != ToolboxWindow.EDITMODE_CONTOUR: return

        current_name = self.getCurrentROIName()
        current_subroi = self.getCurrentSubroiNumber()
        slice_number = int(self.curImage)

        for key_tuple, roi in self.roiManager.all_rois(image_number=slice_number):
            name = key_tuple[0]
            subroi = key_tuple[2]
            if name == current_name:
                if subroi == current_subroi:
                    self.activeRoiPainter.add_roi(roi)
                else:
                    self.sameRoiPainter.add_roi(roi)
            else:
                self.otherRoiPainter.add_roi(roi)

    def drawContours(self):
        """ Plot the contours for the current figure """
        # frame = inspect.getouterframes(inspect.currentframe(), 2)
        # for info in frame:
        #     print("Trace", info[3])
        #     print("Trace", info[3])add
        self.activeRoiPainter.recalculate_patches() # recalculate the position of the active ROI
        self.activeRoiPainter.draw(self.axes, False)
        self.otherRoiPainter.draw(self.axes, False)
        self.sameRoiPainter.draw(self.axes, False)

    def drawSubregion(self):
        """ Plot the autosegment subregion for the current figure """
        if not self.toolbox_window.get_subregion_restriction():
            self.removeSubregion()
            return

        subregion = self.getCurrentSubregion()

        if not self.region_rectangle:
            self.region_rectangle = Rectangle((subregion[1], subregion[0]), subregion[3], subregion[2],
                                              fill=False,
                                              edgecolor='green',
                                              linewidth=2)
            self.axes.add_patch(self.region_rectangle)
        else:
            self.region_rectangle.set_xy((subregion[1], subregion[0]))
            self.region_rectangle.set_width(subregion[3])
            self.region_rectangle.set_height(subregion[2])
            self.region_rectangle.set_visible(True)
        self.axes.draw_artist(self.region_rectangle)


    # convert a single slice to ROIs
    def maskToRois2D(self, name, mask, imIndex, refresh = True):
        if not self.roiManager: return
        self.roiManager.set_mask(name, imIndex, mask)
        if refresh:
            self.updateRoiList()
            self.redraw()

    # convert a 2D mask, a 3D dataset or a 4D (time-resolved) dataset to rois
    def masksToRois(self, maskDict, imIndex):
        for name, mask in maskDict.items():
            if len(mask.shape) == 4: # time-resolved mask
                if self.has_time_dimension():
                    n_frames = min(mask.shape[3], self.n_timepoints)
                    for t in range(n_frames):
                        for sl in range(mask.shape[2]):
                            self.roiManagers[t].set_mask(name, sl, mask[:, :, sl, t])
                else: # 4D mask on a 3D dataset: only load the first time frame
                    for sl in range(mask.shape[2]):
                        self.maskToRois2D(name, mask[:, :, sl, 0], sl, False)
            elif len(mask.shape) > 2: # multislice
                for sl in range(mask.shape[2]):
                    self.maskToRois2D(name, mask[:,:,sl], sl, False)
            else:
                self.maskToRois2D(name, mask, imIndex, False)
        self.updateRoiList()
        self.redraw()

    def displayImage(self, im, cmap=None, redraw = True):
        self.resetBlitBg()
        self.removeMasks()
        self.removeContours()
        self.removeSubregion()
        ImageShow.displayImage(self, im, cmap, redraw)
        self.updateRoiList()  # set the appropriate (sub)roi list for the current image
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
        self.drawSubregion()
        toolbox_window = getattr(self, 'toolbox_window', None)
        if toolbox_window is not None and toolbox_window.is_3D_viewer_visible() and self.curImage is not None:
            self.displayed_slice_changed.emit(int(self.curImage))
        try:
            self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
        except:
            pass

    ##############################################################################################################
    ###
    ### UI Callbacks
    ###
    ##############################################################################################################

    def reblit(self):
        self.reblit_signal.emit()

    @pyqtSlot()
    def do_reblit(self):
        if self.suppressRedraw: return
        if self.blitBg is None or \
                self.blitXlim != self.axes.get_xlim() or \
                self.blitYlim != self.axes.get_ylim():
            self.removeMasks()
            self.removeContours()
            self.removeSubregion()
            self.redraw()
            return
        self.fig.canvas.restore_region(self.blitBg)
        self.plotAnimators()
        self.fig.canvas.blit(self.fig.bbox)
        self.suppressRedraw = True # avoid nested calls
        self.fig.canvas.flush_events()
        self.suppressRedraw = False

    def plotAnimators(self):
        if self.brush_patch is not None:
            self.axes.draw_artist(self.brush_patch)
        if self.roiManager:
            if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
                self.drawContours()
            elif self.editMode == ToolboxWindow.EDITMODE_MASK:
                self.drawMasks()
            self.drawSubregion()

    def redraw(self):
        self.redraw_signal.emit()

    @pyqtSlot()
    def do_redraw(self):
        #print("Redrawing...")
        if self.suppressRedraw: return
        try:
            self.removeMasks()
        except:
            pass
        try:
            self.removeContours()
        except:
            pass
        try:
            self.brush_patch.remove()
        except:
            pass
        try:
            self.removeSubregion()
        except:
            pass
        self.fig.canvas.draw()
        self.suppressRedraw = True # avoid nested calls
        self.fig.canvas.flush_events()
        #plt.pause(0.00001)
        self.suppressRedraw = False
        self.blitBg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.blitXlim = self.axes.get_xlim()
        self.blitYlim = self.axes.get_ylim()
        self.refreshCB()
        try:
            self.updateContourPainters()
        except:
            pass
        try:
            self.updateMasksFromROIs()
        except:
            pass
        self.reblit()

    @pyqtSlot()
    def refreshCB(self):
        # check if ROIs should be autosaved
        now = datetime.now()
        if (now - self.lastsave).total_seconds() > GlobalConfig['AUTOSAVE_INTERVAL'] and \
                not self.separate_thread_running: # avoid autosave while another thread is running
            self.lastsave = now
            self.saveROIPickle()

        if self.wacom:
            self.get_app().setOverrideCursor(Qt.BlankCursor)
        else:
            self.get_app().setOverrideCursor(Qt.ArrowCursor)

    @pyqtSlot()
    def close_slot(self):
        plt.close(self.fig)
        #self.closeCB(None)

    def closeCB(self, event):
        self.toolbox_window.close()
        self.toolbox_window.viewer3D.real_close()
        if not self.basepath: return
        if self.registrationManager:
            self.registrationManager.pickle_transforms()
        self.saveROIPickle()
        # sys.exit(0)
        QApplication.quit()

    @pyqtSlot()
    def updateBrush(self):
        self.moveBrushPatch(None, True)
        self.reblit()

    def moveBrushPatch(self, event = None, force_update = False):
        """
            moves the brush. Returns True if the brush was moved to a new position
        """
        def remove_brush():
            try:
                self.brush_patch.remove()
                #self.fig.canvas.draw()
            except:
                pass
            self.brush_patch = None

        if not self.getCurrentROIName() or self.editMode != ToolboxWindow.EDITMODE_MASK:
            remove_brush()
            return

        brush_type, brush_size = self.toolbox_window.get_brush()

        try:
            mouseX = event.xdata
            mouseY = event.ydata
        except AttributeError: # event is None
            mouseX = None
            mouseY = None

        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
            brush_color = GlobalConfig['BRUSH_PAINT_COLOR']
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            brush_color = GlobalConfig['BRUSH_ERASE_COLOR']
        else:
            brush_color = None
        if (event is not None and (mouseX is None or mouseY is None)) or brush_color is None:
            remove_brush()
            return False

        if event is not None:
            try:
                oldX = self.moveBrushPatch_oldX  # static variables
                oldY = self.moveBrushPatch_oldY
            except:
                oldX = -1
                oldY = -1

            mouseX = np.round(mouseX)
            mouseY = np.round(mouseY)
            self.moveBrushPatch_oldX = mouseX
            self.moveBrushPatch_oldY = mouseY

            if oldX == mouseX and oldY == mouseY and not force_update:
                return False # only return here if we are not forcing an update

        if brush_type == ToolboxWindow.BRUSH_SQUARE:
            if event is not None:
                xy = (math.floor(mouseX - brush_size / 2) + 0.5, math.floor(mouseY - brush_size / 2) + 0.5)
            else:
                try:
                    xy = self.brush_patch.get_xy()
                except:
                    xy = (0.0,0.0)
            if type(self.brush_patch) != SquareBrush:
                try:
                    self.brush_patch.remove()
                except:
                    pass
                self.brush_patch = SquareBrush(xy, brush_size, brush_size, color=brush_color)
                self.axes.add_patch(self.brush_patch)

            self.brush_patch.set_xy(xy)
            self.brush_patch.set_height(brush_size)
            self.brush_patch.set_width(brush_size)

        elif brush_type == ToolboxWindow.BRUSH_CIRCLE:
            if event is not None:
                center = (math.floor(mouseX), math.floor(mouseY))
            else:
                try:
                    center = self.brush_patch.get_center()
                except:
                    center = (0.0,0.0)

            if type(self.brush_patch) != PixelatedCircleBrush:
                try:
                    self.brush_patch.remove()
                except:
                    pass
                self.brush_patch = PixelatedCircleBrush(center, brush_size, color=brush_color)
                self.axes.add_patch(self.brush_patch)

            self.brush_patch.set_center(center)
            self.brush_patch.set_radius(brush_size)

        self.brush_patch.set_animated(True)
        self.brush_patch.set_color(brush_color)
        #self.do_reblit()
        return True

    def modifyMaskFromBrush(self):
        if not self.brush_patch: return
        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
            paintMask = self.brush_patch.to_mask(self.activeMask.shape)
            if self.toolbox_window.get_intensity_aware():
                np.logical_and(paintMask, self.threshold_mask, out=paintMask)
            np.logical_or(self.activeMask, paintMask, out=self.activeMask)
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            brush_mask = self.brush_patch.to_mask(self.activeMask.shape)
            if self.toolbox_window.get_intensity_aware():
                np.logical_and(brush_mask, self.threshold_mask, out=brush_mask)
            eraseMask = np.logical_not(brush_mask)
            np.logical_and(self.activeMask, eraseMask, out=self.activeMask)
            if self.toolbox_window.get_erase_from_all_rois():
                self.otherMask = self.otherMask*eraseMask
        #self.do_reblit()

    # override from ImageShow
    def mouseMoveCB(self, event):
        self.fig.canvas.activateWindow()
        if (self.getState() == 'MUSCLE' and
                self.toolbox_window.get_edit_button_state() in (ToolboxWindow.ADD_STATE, ToolboxWindow.REMOVE_STATE) and
                self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK and
                self.isCursorNormal() and
                event.button != 2 and
                event.button != 3):
            xy = (event.x, event.y)
            if xy == self.oldMouseXY: return  # reject mouse move events when the mouse doesn't move. From parent
            self.oldMouseXY = xy
            moved_to_new_point = self.moveBrushPatch(event)
            if event.button == 1: # because we are overriding MoveCB, we won't call leftPressCB
                if moved_to_new_point:
                    #print("Moved to new point")
                    self.modifyMaskFromBrush()
            self.reblit()
        else:
            if self.brush_patch:
                try:
                    self.brush_patch.remove()
                except:
                    pass
                self.brush_patch = None
            ImageShow.mouseMoveCB(self, event)

    def leftMoveCB(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.toolbox_window.get_subregion_restriction():
            if self.toolbox_window.get_edit_button_state() == ToolboxWindow.SUBREGION_SET_STATE:
                if not self.subregion_start: return
                start_row = self.subregion_start[0]
                start_col = self.subregion_start[1]

                new_row = int(event.ydata)
                new_col = int(event.xdata)

                new_start_row = min(start_row, new_row)
                new_end_row = max(start_row, new_row)
                new_start_col = min(start_col, new_col)
                new_end_col = max(start_col, new_col)
            elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.SUBREGION_MOVE_STATE:
                delta_row = int(event.ydata) - self.subregion_translate_start[0]
                delta_col = int(event.xdata) - self.subregion_translate_start[1]

                original_subregion = self.subregion_start
                new_start_row = original_subregion[0] + delta_row
                new_start_col = original_subregion[1] + delta_col
                new_end_row = new_start_row + original_subregion[2]
                new_end_col = new_start_col + original_subregion[3]

            if new_start_row < 0:
                new_start_row = 0

            if new_start_col < 0:
                new_start_col = 0

            if new_end_row >= self.image.shape[0]:
                new_end_row = self.image.shape[0]

            if new_end_col >= self.image.shape[1]:
                new_end_col = self.image.shape[1]

            self.setCurrentSubregion(
                (new_start_row, new_start_col, new_end_row - new_start_row, new_end_col - new_start_col))
            self.reblit()
            return


        if self.getState() != 'MUSCLE': return

        roi = self.getCurrentROI()
        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:  # event.key == 'shift' or checkCapsLock():
            self.movePoint(roi, event)
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.TRANSLATE_STATE:
            if self.translateDelta is None: return
            newCenter = (event.xdata - self.translateDelta[0], event.ydata - self.translateDelta[1])
            roi.moveCenterTo(newCenter)
            self.reblit()
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.ROTATE_STATE:
            if self.rotationDelta is None: return
            newAngle = roi.getOrientation( (event.xdata, event.ydata), center = self.rotationDelta[0])
            roi.reorientByAngle(newAngle - self.rotationDelta[1])
            self.reblit()

    def leftPressCB(self, event):
        if not self.imPlot.contains(event):
            return

        # These two are independent on the existance of an active ROI
        if self.toolbox_window.get_subregion_restriction():
            if self.toolbox_window.get_edit_button_state() == ToolboxWindow.SUBREGION_SET_STATE:
                self.subregion_start = (int(event.ydata), int(event.xdata))
                self.removeSubregion()
                self.redraw()
                return

            if self.toolbox_window.get_edit_button_state() == ToolboxWindow.SUBREGION_MOVE_STATE:
                self.subregion_translate_start = (int(event.ydata), int(event.xdata))
                self.subregion_start = self.getCurrentSubregion()
                self.removeSubregion()
                self.redraw()
                return

        # Only set if there is an active ROI
        if self.getState() != 'MUSCLE': return

        if self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK:
            if self.toolbox_window.get_intensity_aware():
                # if the operation has to be intensity aware, create a threshold mask based on current point
                intensity = self.image[int(event.ydata), int(event.xdata)]
                threshold_intensity = intensity * self.toolbox_window.get_intensity_threshold()
                lower_threshold = intensity - threshold_intensity
                upper_threshold = intensity + threshold_intensity
                self.threshold_mask = self.image < upper_threshold
                np.logical_and(self.threshold_mask, self.image > lower_threshold, out=self.threshold_mask)
            self.modifyMaskFromBrush()
        else:
            #print("Edit button state", self.toolbox_window.get_edit_button_state())
            roi = self.getCurrentROI()
            knotIndex, knot = roi.findKnotEvent(event)
            if self.toolbox_window.get_edit_button_state() == ToolboxWindow.TRANSLATE_STATE:
                center = roi.getCenterOfMass()
                if center is None:
                    self.translateDelta = None
                    return
                self.saveSnapshot()
                self.translateDelta = (event.xdata - center[0], event.ydata - center[1])
            elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.ROTATE_STATE:
                center = roi.getCenterOfMass()
                if center is None:
                    self.rotationDelta = None
                    return
                self.saveSnapshot()
                startAngle = roi.getOrientation(center=center)
                self.rotationDelta = (center, roi.getOrientation( (event.xdata, event.ydata), center=center ) - startAngle)
            elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
                if knotIndex is not None:
                    self.saveSnapshot()
                    roi.removeKnot(knotIndex)
                    self.reblit()
            elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
                self.saveSnapshot()
                if knotIndex is None:
                    self.addPoint(roi, event)
                else:
                    self.currentPoint = knotIndex

    def leftReleaseCB(self, event):
        self.currentPoint = None  # reset the state
        self.translateDelta = None
        self.rotationDelta = None
        self.subregion_start = None
        self.suregion_translate_start = None
        if self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.saveSnapshot() # save state before modification
            if self.roiManager is not None:
                self.roiManager.set_mask(self.getCurrentROIName(), self.curImage, self.activeMask)
            if self.toolbox_window.get_erase_from_all_rois():
                other_mask_bool = self.otherMask.astype(np.bool)
                for (key_tuple, mask) in self.roiManager.all_masks(image_number=self.curImage):
                    if key_tuple[0] == self.getCurrentROIName(): continue
                    self.roiManager.set_mask(key_tuple[0], key_tuple[1], np.logical_and(mask, other_mask_bool))
        self.emit_mask_slice_changed()

    def rightPressCB(self, event):
        self.hideRois = GlobalConfig['HIDE_ROIS_RIGHTCLICK']
        self.redraw()

    def rightReleaseCB(self, event):
        self.hideRois = False
        self.redraw()

    def mouseScrollCB(self, event):
        modifier_status, *_ = self.get_key_modifiers(event)
        if modifier_status['ctrl']:
            if event.step < 0:
                self.reduce_brush_size.emit()
            elif event.step > 0:
                self.increase_brush_size.emit()
            return
        ImageShow.mouseScrollCB(self, event)

    @staticmethod
    def get_key_modifiers(event):
        modifiers = event.guiEvent.modifiers()
        try:
            pressed_key_without_modifiers = event.key.split('+')[-1]  # this gets the nonmodifier key if the pressed key is ctrl+z for example
        except:
            pressed_key_without_modifiers = ''
        is_key_modifier_only = (pressed_key_without_modifiers in ['shift', 'control', 'ctrl', 'cmd', 'super', 'alt'])
        out_modifiers = {'ctrl': (modifiers & (Qt.ControlModifier | Qt.MetaModifier)) != Qt.NoModifier,
                         'shift': (modifiers & Qt.ShiftModifier) == Qt.ShiftModifier,
                         'alt': (modifiers & Qt.AltModifier) == Qt.AltModifier,
                         'none': (modifiers == Qt.NoModifier)}
        return out_modifiers, is_key_modifier_only, pressed_key_without_modifiers


    def keyPressCB(self, event):
        modifier_status, is_key_modifier_only, pressed_key_without_modifiers = self.get_key_modifiers(event)

        if is_key_modifier_only:
            if modifier_status['shift']:
                self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.ADD_STATE)
            elif modifier_status['ctrl']:
                self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.REMOVE_STATE)
            return

        if modifier_status['ctrl']:
            if pressed_key_without_modifiers in self.shortcuts:
                self.shortcuts[pressed_key_without_modifiers]()
            return

        if event.key == 'n':
            if self.registration_available:
                self.propagate()
        elif event.key == 'b':
            if self.registration_available:
                self.propagateBack()
        elif event.key == '-' or event.key == 'y' or event.key == 'z':
            self.reduce_brush_size.emit()
        elif event.key == '+' or event.key == 'x':
            self.increase_brush_size.emit()
        elif event.key == 'r':
            self.roiRemoveOverlap()
        elif event.key in (',', 'shift+left'):
            self.previous_timepoint()
        elif event.key in ('.', 'shift+right'):
            self.next_timepoint()
        else:
            ImageShow.keyPressCB(self, event)

    def keyReleaseCB(self, event):
        modifier_status, is_key_modifier_only, pressed_key_without_modifiers = self.get_key_modifiers(event)

        if modifier_status['shift']:
            self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.ADD_STATE)
        elif modifier_status['ctrl']:
            self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.REMOVE_STATE)
        else:
            self.toolbox_window.restore_edit_button_state()


    ################################################################################################################
    ###
    ### I/O
    ###
    ################################################################################################################

    def getDatasetAsNumpy(self):
        return np.transpose(np.stack(self.imList), [1,2,0])

    @pyqtSlot(str)
    def saveROIPickle(self, roiPickleName=None, async_write = False):

        @separate_thread_decorator
        def write_file(name, bytes_to_write):
            with open(name, 'wb') as f:
                f.write(bytes_to_write)

        showWarning = True
        if not roiPickleName:
            roiPickleName = self.getRoiFileName()
            showWarning = False # don't show a empty roi warning if autosaving
            async_write = True

        #print("Saving ROIs", roiPickleName)
        if self.has_time_dimension():
            rois_not_empty = any(manager is not None and not manager.is_empty()
                                 for manager in self.roiManagers.values())
        else:
            rois_not_empty = self.roiManager is not None and not self.roiManager.is_empty()

        if rois_not_empty:  # make sure ROIs are not empty
            # 'roiManager' always holds the current-timepoint manager, so that older Dafne
            # versions can still open ROI files saved from a time-resolved dataset
            dumpObj = {'classifications': self.classifications,
                       'roiManager': self.roiManager }
            if self.has_time_dimension():
                dumpObj['roiManagers'] = self.roiManagers
                dumpObj['currentTimepoint'] = self.current_timepoint
            if async_write:
                bytes_to_write = pickle.dumps(dumpObj)
                write_file(roiPickleName, bytes_to_write) # write file asynchronously for a smoother experience in autosave
            else:
                pickle.dump(dumpObj, open(roiPickleName, 'wb'))
        else:
            if showWarning: self.alert('ROIs are empty - not saved')

    @pyqtSlot(str)
    def loadROIPickle(self, roiPickleName=None):
        if not roiPickleName:
            roiPickleName = self.getRoiFileName()
        #print("Loading ROIs", roiPickleName)
        try:
            dumpObj = pickle.load(open(roiPickleName, 'rb'))
        except UnicodeDecodeError:
            print('Warning: Unicode decode error')
            dumpObj = pickle.load(open(roiPickleName, 'rb'), encoding='latin1')
        except:
            traceback.print_exc()
            self.alert("Unspecified error", "Error")
            return

        roiManager = None
        savedRoiManagers = None
        classifications = self.classifications

        if isinstance(dumpObj, (ROIManager, utils.ROIManager.ROIManager)):
            roiManager = dumpObj
        elif isinstance(dumpObj, dict):
            try:
                classifications = dumpObj['classifications']
                roiManager = dumpObj['roiManager']
            except KeyError:
                self.alert("Unrecognized saved ROI type")
                return
            savedRoiManagers = dumpObj.get('roiManagers', None)

        def validate_roi_manager(manager):
            if not isinstance(manager, (ROIManager, utils.ROIManager.ROIManager)):
                return False
            if manager.mask_size[0] != self.image.shape[0] or \
                manager.mask_size[1] != self.image.shape[1]:
                return False
            # compatibility with old versions that don't have autosegment subregions
            try:
                manager.autosegment_subregions
            except AttributeError:
                manager.autosegment_subregions = {}
            return True

        if not isinstance(roiManager, (ROIManager, utils.ROIManager.ROIManager)):
            self.alert("Unrecognized saved ROI type")
            return

        if not validate_roi_manager(roiManager):
            self.alert("ROI for wrong dataset")
            return

        #print('Rois loaded')
        self.clearAllROIs()
        if self.has_time_dimension():
            newRoiManagers = {}
            for t in range(self.n_timepoints):
                manager = savedRoiManagers.get(t, None) if savedRoiManagers else None
                if manager is None or not validate_roi_manager(manager):
                    manager = ROIManager(self.imList[0].shape)
                newRoiManagers[t] = manager
            if savedRoiManagers is None:
                # a 3D ROI file loaded on a time-resolved dataset: load it into the current frame
                newRoiManagers[self.current_timepoint] = roiManager
            self.roiManagers = newRoiManagers
            self.roiManager = self.roiManagers[self.current_timepoint]
        else:
            self.roiManager = roiManager
            if self.roiManagers:
                self.roiManagers[self.current_timepoint] = roiManager
        available_classes = self.toolbox_window.get_available_classes()
        for i, classification in enumerate(classifications[:]):
            if classification not in available_classes:
                classifications[i] = 'None'

        self.classifications = classifications
        self.updateRoiList()
        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
        self.redraw()

    def _load_volume_from_path(self, path):
        # Replicates ImageShow.loadDirectory, but splits 4D (time-resolved) volumes into
        # separate time frames before the first image is displayed
        medical_volume, affine_valid, title, basepath, basename = dosma_volume_from_path(path, self.fig.canvas)

        if medical_volume.volume.ndim > 4:
            reduced_data = reduce_array_dimensions(medical_volume.volume, self.fig.canvas)
            if reduced_data is None:
                raise ValueError('Loading cancelled by user')
            medical_volume = MedicalVolume(reduced_data, medical_volume.affine)

        self.imList = []
        self.dicomHeaderList = None
        self.medical_volume = None
        self.affine = None
        self.resolution_valid = False
        self.resolution = [1, 1, 1]

        self.load_dosma_volume(medical_volume)
        self.resolution_valid = affine_valid
        self.basename = basename
        self.basepath = basepath
        self.fig.canvas.manager.set_window_title(title)

        self._detect_dicom_time_series()
        self._split_time_frames()

        if len(self.imList) > 0:
            try:
                self.imPlot.remove()
            except:
                pass
            self.imPlot = None
            self.curImage = 0
            self.displayImage(int(0))
            self.axes.set_xlim(-0.5, self.image.shape[1] - 0.5)
            self.axes.set_ylim(self.image.shape[0] - 0.5, -0.5)

    def _restore_bundle_contrasts(self, contrast_arrays, contrast_names):
        """ Restore the additional contrasts saved in a bundle as 'data2', 'data3', ...
            A 4D array becomes a time-resolved contrast, a 3D one a static contrast. """
        if not contrast_arrays:
            return
        affine = self.affine if self.affine is not None else np.eye(4)
        for list_index, key_index in enumerate(sorted(contrast_arrays)):
            data = contrast_arrays[key_index]
            try:
                name = contrast_names[list_index]
            except (TypeError, IndexError):
                name = f'Contrast {key_index}'
            # avoid collisions with the base contrast label or already-restored names
            if name == ToolboxWindow.BASE_CONTRAST_LABEL or name in self.additional_contrasts:
                name = f'{name} ({key_index})'
            if data.ndim == 4:
                if not self.has_time_dimension() or data.shape[3] != self.n_timepoints or \
                        data.shape[:3] != tuple(self.medical_volume.shape):
                    print(f'Skipping bundle contrast {name}: does not match the dataset')
                    continue
                contrast_frames = [MedicalVolume(np.ascontiguousarray(data[..., t]), affine)
                                   for t in range(data.shape[3])]
                self.additional_contrast_frames[name] = contrast_frames
                self.additional_contrasts[name] = contrast_frames[self.current_timepoint]
            else:
                if tuple(data.shape) != tuple(self.medical_volume.shape):
                    print(f'Skipping bundle contrast {name}: does not match the dataset')
                    continue
                self.additional_contrasts[name] = MedicalVolume(data, affine)
            self.toolbox_window.add_contrast_to_combo(name)

    @pyqtSlot(str, str)
    @pyqtSlot(str)
    def loadDirectory(self, path, override_class=None):
        self.setSplash(True, 0, 1, "Loading dataset")

        def __reset_state():
            self.imList = []
            self.resetInternalState()
            self.override_class = override_class
            self.resolution_valid = False
            self.original_affine=None
            self.affine = None
            self.original_headers=None
            self.image = None
            self.resolution = [1, 1, 1]

        def __cleanup():
            __reset_state()
            self.setSplash(False)

        def __error(error = None):
            print(error, file=sys.stderr)
            self.alert("Error loading dataset. See the log for details", "Error")
            __cleanup()
            self.displayImage(None)
            self.redraw()

        __reset_state()
        _, ext = os.path.splitext(path)
        mask_dictionary = None
        bundle_contrast_arrays = {}
        bundle_contrast_names = None
        if ext.lower() == '.npz':
            # data and mask bundle
            bundle = np.load(path, allow_pickle=False)
            if 'data' not in bundle and 'image' not in bundle:
                self.alert('No data in bundle!', 'Error')
                self.setSplash(False, 1, 2, "")
                return
            if 'comment' in bundle:
                self.alert('Loading bundle with comment:\n' + str(bundle['comment']), 'Info')

            self.basepath = os.path.dirname(path)
            try:
                if 'data' in bundle:
                    base_data = bundle['data']
                elif 'image' in bundle:
                    base_data = bundle['image']
                else:
                    __error('No data in bundle!') # should never happen because we are checking above
                    return
                base_data = reduce_array_dimensions(base_data, self.fig.canvas)
                if base_data is None:
                    __cleanup()
                    return
                self.loadNumpyArray(base_data)
            except Exception as e:
                __error(e)
                return

            if 'affine' in bundle:
                self.affine = bundle['affine']
                self.medical_volume._affine = self.affine

            if 'resolution' in bundle:
                self.resolution = list(bundle['resolution'])
                if len(self.resolution) == 2:
                    self.resolution.append(1.0)
                self.resolution_valid = True
                print('Resolution', self.resolution)
                if self.affine is None:
                    self.affine = np.diag(self.resolution + [1])
                    self.medical_volume._affine = self.affine

            mask_dictionary = {}
            for key in bundle:
                if key.startswith('mask_'):
                    mask_name = key[len('mask_'):]
                    mask_dictionary[mask_name] = bundle[key]
                    print('Found mask', mask_name)
                else:
                    contrast_match = re.fullmatch(r'data(\d+)', key)
                    if contrast_match:
                        bundle_contrast_arrays[int(contrast_match.group(1))] = bundle[key]
                        print('Found additional contrast', key)
            if 'contrast_names' in bundle:
                bundle_contrast_names = [str(contrast_name) for contrast_name in bundle['contrast_names']]

            self._split_time_frames() # handle 4D (time-resolved) bundles

            # from the parent class
            try:
                self.imPlot.remove()
            except:
                pass
            self.imPlot = None
            self.curImage = 0
            self.displayImage(int(0))
            self.axes.set_xlim(-0.5, self.image.shape[1] - 0.5)
            self.axes.set_ylim(self.image.shape[0] - 0.5, -0.5)
        else:
            try:
                self._load_volume_from_path(path)
                if path.endswith(".nii.gz")| path.endswith(".nii"):
                    # load original informations
                    original_volume = nib.load(path)
                    self.original_affine = original_volume.affine
                    self.original_headers = original_volume.header
            except Exception as e:
                __error(e)
                return

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
                self.axes.set_aspect(aspect=self.resolution[0]/self.resolution[1])
                self.medical_volume._affine = np.diag(self.resolution + [1])
                for time_frame in self.time_frames:
                    time_frame._affine = self.medical_volume._affine

        # this is in case appendimage was never called
        if len(self.classifications) == 0:
            self.update_all_classifications()

        self.additional_contrasts[ToolboxWindow.BASE_CONTRAST_LABEL] = self.medical_volume
        self._restore_bundle_contrasts(bundle_contrast_arrays, bundle_contrast_names)

        roi_bak_name = self.getRoiFileName() + '.' + datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            shutil.copyfile(self.getRoiFileName(), roi_bak_name)
        except:
            print("Warning: cannot copy roi file")

        self._setup_timepoint_managers()
        #self.loadROIPickle()
        self.updateRoiList()
        try:
            self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
        except:
            pass
        self.redraw()
        self.toolbox_window.general_enable(True)
        self.toolbox_window.set_exports_enabled(numpy= True,
                                                dicom= (self.dicomHeaderList is not None),
                                                nifti= (self.affine is not None)
                                                )
        if mask_dictionary:
            self.setSplash(True, 1, 2, "Loading masks")
            self.masksToRois(mask_dictionary, 0)
        self.setSplash(False, 1, 2, "Loading masks")
        self.emit_viewer3d_data()

    def update_all_classifications(self):
        self.classifications = []
        for imIndex in range(len(self.imList)):
            if self.override_class:
                self.classifications.append(self.override_class)
                continue
            if not self.dl_classifier:
                self.classifications.append('None')
                continue
            class_input = {'image': self.imList[imIndex], 'resolution': self.resolution[0:2]}
            class_str = self.dl_classifier(class_input)
            # class_str = 'Thigh' # DEBUG
            print("Classification", class_str)
            self.classifications.append(class_str)


    def appendImage(self, im):
        ImageShow.appendImage(self, im)
        if self.override_class:
            self.classifications.append(self.override_class)
            return
        if not self.dl_classifier:
            self.classifications.append('None')
            return
        class_input = {'image': self.imList[-1], 'resolution': self.resolution[0:2]}
        class_str = self.dl_classifier(class_input)
        #class_str = 'Thigh' # DEBUG
        print("Classification", class_str)
        self.classifications.append(class_str)

    @pyqtSlot(str, str)
    @separate_thread_decorator
    def saveBundle(self, path_out: str, comment: str):
        self.setSplash(True, 0, 1, "Saving bundle...")

        if self._is_current_model_3D():
            
            self.setSplash(True, 0, 4, "Calculating maps...")
            saved_files_count = self.count_saved_files()
            print(f"saved_files_count {saved_files_count}")
            current_model = self.classifications[int(self.curImage)]
            total_next_indices = len(self.incrLearnDataTrain.get(current_model, {}))

            if saved_files_count > total_next_indices:
                self.load_saved_npz()
                total_next_indices = len(self.incrLearnDataTrain.get(current_model, {}))
                print(f"files count {total_next_indices}")
            
            allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)   
            next_index, key, value_image = self.add_to_incrLearnData(self.incrLearnDataTrain, dataForTraining) 
            next_index, key, value_segm = self.add_to_incrLearnData(self.incrLearnSegTrain, segForTraining) 
            
            if key not in self.incrLearnMeanDice:
                self.incrLearnMeanDice[key] = {} 
                self.incrLearnMeanDice[key][next_index] = meanDiceScore
            else:
                self.incrLearnMeanDice[key][next_index] = meanDiceScore

            affine_transform = self.original_affine 
            # for incremental learning
            if key not in self.incrementalLearningAffine:
                self.incrementalLearningAffine[key] = {} 
                self.incrementalLearningAffine[key][next_index] = self.affine
            else:
                self.incrementalLearningAffine[key][next_index] = self.affine
            
            bundle = self.prepare_numpy_bundle_IL_3D(value_image, value_segm, meanDiceScore, key, comment)
            np.savez_compressed(path_out, **bundle)

            if GlobalConfig['DO_INCREMENTAL_LEARNING']:

                bundle = self.prepare_numpy_bundle_IL_3D(value_image, value_segm, meanDiceScore, key)
                directory=os.path.join(GlobalConfig['NUMPY_FILE_3D'], self.classifications[int(self.curImage)])

                if not os.path.isdir(directory):
                    os.makedirs(directory)

                np.savez_compressed(os.path.join(directory, f'temp_{next_index}'), **bundle)
                
        else:
            bundle = self.prepare_numpy_bundle(comment)
            np.savez_compressed(path_out, **bundle)
        self.setSplash(False)
    
    def add_to_incrLearnData(self, incrLearnData, data):
        for key, value in data.items():
            if key not in incrLearnData:
                incrLearnData[key] = {}  # create the dict if it not exists
                next_index = 0  # start from 0 for the first time
            else:
                next_index = max(incrLearnData[key].keys(), default=-1) + 1  # find the next index

            incrLearnData[key][next_index] = value  # add the array with the new index
        return next_index, key, value
    
    def count_saved_files(self):
        """Count the number of files saved in the temporary directory."""
        directory=os.path.join(GlobalConfig['NUMPY_FILE_3D'], self.classifications[int(self.curImage)])
        if not os.path.isdir(directory):
            os.makedirs(directory)
        folder_path = os.listdir(directory)
        return len([name for name in folder_path if name.endswith('.npz')])
    
    def load_saved_npz(self):
        """Load the data from saved NPZ files."""
        class_str = self.classifications[int(self.curImage)]
        try:
            folder_path = os.listdir(os.path.join(GlobalConfig['NUMPY_FILE_3D'], class_str))
            temp_files = sorted([f for f in folder_path if f.startswith('temp_') and f.endswith('.npz')],
                                key=lambda x: int(x.split('_')[1].split('.npz')[0]))
            
            for file in temp_files:
                file_path = os.path.join(GlobalConfig['NUMPY_FILE_3D'], class_str, file)
                data = np.load(file_path, allow_pickle=True)

                key = class_str

                if key not in self.incrLearnDataTrain:
                    self.incrLearnDataTrain[key] = {}
                    self.incrLearnSegTrain[key] = {}
                    self.incrementalLearningAffine[key] = {}
                    self.incrLearnMeanDice[key] = {}
                    next_index = 0
                else:
                    # the index is numeric, so get the maximum index and add 1 for the next index
                    next_index = max(self.incrLearnDataTrain[key].keys(), default=-1) + 1

                contrast_values = OrderedDict()
                for file_key in data.files:
                    if file_key == 'data':
                        contrast_values['image'] = data[file_key]
                    elif re.fullmatch(r'data\d+', file_key):
                        contrast_values[f'image{file_key[len("data"):]}'] = data[file_key]
                self.incrLearnDataTrain[key][next_index] = contrast_values
                self.incrementalLearningAffine[key][next_index] = data['affine']
                self.incrLearnMeanDice[key][next_index] = data['dice']

                for sub_key in [k for k in data.files if k.startswith('mask_')]:

                    sub_key = str(sub_key)
                    original_key = sub_key.replace("mask_", "")
                    self.incrLearnSegTrain[key][next_index] = {}
                    self.incrLearnSegTrain[key][next_index][original_key] = data[sub_key]
            
            print("Data from NPZ files loaded.")
        except Exception as e:
            print(f"Error loading NPZ files: {e}")

    def save_3D_bundle_for_IL(self, dataForTraining, segForTraining, meanDiceScore, set_splash=False, force_save=False):
        if not force_save and self.bundle_saved_for_IL:
            return 0
        self.bundle_saved_for_IL = True
        next_index, key, value_image = self.add_to_incrLearnData(self.incrLearnDataTrain, dataForTraining)
        next_index, key, value_segm = self.add_to_incrLearnData(self.incrLearnSegTrain, segForTraining)

        if key not in self.incrLearnMeanDice:
            self.incrLearnMeanDice[key] = {}
            self.incrLearnMeanDice[key][next_index] = meanDiceScore
        else:
            self.incrLearnMeanDice[key][next_index] = meanDiceScore

        if set_splash:
            self.setSplash(True, 3, 4, "Saving file...")

        affine_transform = self.original_affine
        # for incremental learning
        if key not in self.incrementalLearningAffine:
            self.incrementalLearningAffine[key] = {}
            self.incrementalLearningAffine[key][next_index] = self.affine
        else:
            self.incrementalLearningAffine[key][next_index] = self.affine

        if GlobalConfig['DO_INCREMENTAL_LEARNING']:
            # saving as bundle
            if set_splash:
                self.setSplash(True, 0, 1, "Saving bundle...")
            bundle = self.prepare_numpy_bundle_IL_3D(value_image, value_segm, meanDiceScore, key)
            directory = os.path.join(GlobalConfig['NUMPY_FILE_3D'], self.classifications[int(self.curImage)])

            if not os.path.isdir(directory):
                os.makedirs(directory)

            np.savez_compressed(os.path.join(directory, f'temp_{next_index}'), **bundle)
            if set_splash:
                self.setSplash(False)

        return next_index

    def update_3D_incrLearn_objects(self):
        saved_files_count = self.count_saved_files()
        print(f"saved_files_count {saved_files_count}")
        current_model = self.classifications[int(self.curImage)]
        total_next_indices = len(self.incrLearnDataTrain.get(current_model, {}))

        if saved_files_count > total_next_indices:
            self.load_saved_npz()
            total_next_indices = len(self.incrLearnDataTrain.get(current_model, {}))
            print(f"files count {total_next_indices}")

    def _get_all_frames_masks(self):
        """ Return dict roi_name -> 4D mask (H, W, slices, timepoints) with the masks of every
            time frame. """
        masks = {}
        n_slices = len(self.imList)
        mask_shape = (self.image.shape[0], self.image.shape[1], n_slices, self.n_timepoints)
        for roi_name in self._selected_roi_names(True):
            mask_4d = np.zeros(mask_shape, dtype=np.uint8)
            for t in range(self.n_timepoints):
                manager = self.roiManagers[t]
                for z in range(n_slices):
                    if manager.contains(roi_name, z):
                        mask_4d[:, :, z, t] = manager.get_mask(roi_name, z)
            masks[roi_name] = mask_4d
        return masks

    @pyqtSlot(str, str, bool)
    @separate_thread_decorator
    def saveResults(self, pathOut: str, outputType: str, single_frame: bool = False):
        # outputType is 'dicom', 'npy', 'npz', 'nifti', 'compact_dicom', 'compact_nifti'
        print("Saving results...")

        if self.has_time_dimension() and not single_frame:
            # export the masks of all the time frames as 4D arrays. This is a pure export:
            # no incremental-learning bookkeeping, which is frame-based
            self.setSplash(True, 0, 1, "Saving masks...")
            allMasks = self._get_all_frames_masks()
            if outputType == 'nifti':
                save_nifti_masks(pathOut, allMasks, self.affine)
            elif outputType == 'npy':
                save_npy_masks(pathOut, allMasks, self.affine)
            elif outputType == 'compact_nifti':
                save_single_nifti(pathOut, allMasks, self.affine)
            else: # assume the most generic outputType == 'npz':
                save_npz_masks(pathOut, allMasks, self.affine)
            self.setSplash(False, 1, 1, "End")
            return

        self.setSplash(True, 0, 4, "Calculating maps...")

        if self._is_current_model_3D():
            self.update_3D_incrLearn_objects()

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        if self._is_current_model_3D():
            next_index = self.save_3D_bundle_for_IL(dataForTraining, segForTraining, meanDiceScore, set_splash=True, force_save=True)

        if outputType == 'dicom':
            save_dicom_masks(pathOut, allMasks, self.affine, self.dicomHeaderList)
        elif outputType == 'nifti':
            if self._is_current_model_3D():
                save_nifti_masks_3D(pathOut, next_index, allMasks, self.original_affine, self.affine, self.original_headers)
            else:
                save_nifti_masks(pathOut, allMasks, self.affine)
        elif outputType == 'npy':
            save_npy_masks(pathOut, allMasks, self.affine)
        elif outputType == 'compact_dicom':
            save_single_dicom_dataset(pathOut, allMasks, self.affine, self.dicomHeaderList)
        elif outputType == 'compact_nifti':
            save_single_nifti(pathOut, allMasks, self.affine)
        else: # assume the most generic outputType == 'npz':
            save_npz_masks(pathOut, allMasks, self.affine)

        # perform incremental learning
        if GlobalConfig['DO_INCREMENTAL_LEARNING']:
            if self._is_current_model_3D():
                # Ask for confirmation as it can take a long time
                answer = self.question('Do you want to perform incremental learning with the new data? This can take a long time')
                if answer:
                    self.setSplash(True, 1, 4, "Incremental learning...")
                    self.incrementalLearn_3D(self.incrLearnDataTrain, self.incrLearnSegTrain, self.incrementalLearningAffine, self.incrLearnMeanDice, True)
            else:
                self.setSplash(True, 1, 4, "Incremental learning...")
                self.incrementalLearn(dataForTraining, segForTraining, meanDiceScore, True)

        self.setSplash(False, 4, 4, "End")

    @pyqtSlot(str)
    @separate_thread_decorator
    def saveStats_singleslice(self, file_out: str):
        """ Saves the statistics for a datasets. Exported statistics:
            - Number of slices where ROI is present
            - Number of voxels
            - Average value of the data over ROI
            - Standard Deviation of the data
            - 0-25-50-75-100 percentiles of the data distribution
        """
        self.setSplash(True, 0, 2, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        self.setSplash(True, 1, 2, "Calculating stats...")

        dataset = self.getDatasetAsNumpy()

        csv_file = open(file_out, 'w')
        field_names = ['roi_name',
                       'slice',
                       'voxels',
                       'volume',
                       'mean',
                       'standard_deviation',
                       'perc_0',
                       'perc_25',
                       'perc_50',
                       'perc_75',
                       'perc_100']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()

        for roi_name, roi_mask in allMasks.items():
            for slice_number in range(roi_mask.shape[2]):
                mask_slice = roi_mask[:, :, slice_number]
                data_slice = dataset[:, :, slice_number]
                if mask_slice.sum() == 0:
                    continue
                try:
                    csvRow = {}
                    csvRow['roi_name'] = roi_name
                    csvRow['slice'] = slice_number
                    mask = mask_slice > 0
                    masked = np.ma.array(data_slice, mask=np.logical_not(mask))
                    csvRow['voxels'] = mask.sum()
                    try:
                        csvRow['volume'] = csvRow['voxels']*self.resolution[0]*self.resolution[1]*self.resolution[2]
                    except:
                        csvRow['volume'] = 0
                    compressed_array = masked.compressed()
                    csvRow['mean'] = compressed_array.mean()
                    csvRow['standard_deviation'] = compressed_array.std()
                    csvRow['perc_0'] = compressed_array.min()
                    csvRow['perc_100'] = compressed_array.max()
                    csvRow['perc_25'] = np.percentile(compressed_array, 25)
                    csvRow['perc_50'] = np.percentile(compressed_array, 50)
                    csvRow['perc_75'] = np.percentile(compressed_array, 75)
                    csv_writer.writerow(csvRow)
                except:
                    print('Error calculating statistics for ROI', roi_name)
                    traceback.print_exc()

        csv_file.close()
        self.setSplash(False, 2, 2, "Finished")


    @pyqtSlot(str)
    @separate_thread_decorator
    def saveStats(self, file_out: str):
        """ Saves the statistics for a datasets. Exported statistics:
            - Number of slices where ROI is present
            - Number of voxels
            - Average value of the data over ROI
            - Standard Deviation of the data
            - 0-25-50-75-100 percentiles of the data distribution
        """
        self.setSplash(True, 0, 2, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        self.setSplash(True, 1, 2, "Calculating stats...")

        dataset = self.getDatasetAsNumpy()

        csv_file = open(file_out, 'w')
        field_names = ['roi_name',
                       'slices',
                       'voxels',
                       'volume',
                       'mean',
                       'standard_deviation',
                       'perc_0',
                       'perc_25',
                       'perc_50',
                       'perc_75',
                       'perc_100']
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()

        for roi_name, roi_mask in allMasks.items():
            try:
                csvRow = {}
                csvRow['roi_name'] = roi_name
                mask = roi_mask > 0
                masked = np.ma.array(dataset, mask=np.logical_not(roi_mask))
                csvRow['voxels'] = mask.sum()
                try:
                    csvRow['volume'] = csvRow['voxels']*self.resolution[0]*self.resolution[1]*self.resolution[2]
                except:
                    csvRow['volume'] = 0
                # count the slices where the roi is present
                mask_pencil = np.sum(mask, axis=(0,1))
                csvRow['slices'] = np.sum(mask_pencil > 0)
                compressed_array = masked.compressed()
                csvRow['mean'] = compressed_array.mean()
                csvRow['standard_deviation'] = compressed_array.std()
                csvRow['perc_0'] = compressed_array.min()
                csvRow['perc_100'] = compressed_array.max()
                csvRow['perc_25'] = np.percentile(compressed_array, 25)
                csvRow['perc_50'] = np.percentile(compressed_array, 50)
                csvRow['perc_75'] = np.percentile(compressed_array, 75)
                csv_writer.writerow(csvRow)
            except:
                print('Error calculating statistics for ROI', roi_name)
                traceback.print_exc()

        csv_file.close()
        self.setSplash(False, 2, 2, "Finished")

    @pyqtSlot(str, bool, int, int)
    @separate_thread_decorator
    def saveRadiomics(self, file_out: str, do_quantization=True, quant_levels=32, erode_px=0):
        """ Saves the radiomics features from pyradiomics
        """
        self.setSplash(True, 0, 2, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        self.setSplash(True, 1, 2, "Calculating stats...")

        dataset = self.getDatasetAsNumpy()

        if do_quantization:
            data_min = dataset.min()
            data_max = dataset.max()
            dataset = np.round((dataset-data_min) * quant_levels / (data_max - data_min))

        first_run = True
        header = 'roi_name'

        with open(file_out, 'w') as featureFile:
            for roi_name, roi_mask in allMasks.items():
                if erode_px > 0:
                    eroded_mask = binary_erosion(roi_mask, iterations=erode_px)
                else:
                    eroded_mask = roi_mask

                extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
                image = sitk.GetImageFromArray(dataset)
                features = extractor.execute(image, sitk.GetImageFromArray(eroded_mask.astype(np.uint8)))
                featureLine = f'{roi_name}'
                for k, v in features.items():
                    if k.startswith('original'):
                        if first_run:
                            header += ',' + k
                        try:
                            featureLine += ',{:.6f}'.format(v[0])
                        except:
                            featureLine += ',{:.6f}'.format(v)
                if first_run:
                    featureFile.write(header + '\n')
                    first_run = False
                featureFile.write(featureLine + '\n')

        self.setSplash(False, 2, 2, "Finished")

    def _contrast_data_for_bundle(self, contrast_name):
        """ Numpy data of a contrast for bundle export: 4D (time frames stacked on the 4th
            axis) for time-resolved volumes, 3D otherwise (including contrasts that are
            static over a time-resolved dataset). """
        if self.has_time_dimension():
            if contrast_name == ToolboxWindow.BASE_CONTRAST_LABEL:
                return np.stack([frame.volume for frame in self.time_frames], axis=-1)
            if contrast_name in self.additional_contrast_frames:
                return np.stack([frame.volume
                                 for frame in self.additional_contrast_frames[contrast_name]], axis=-1)
        return self.additional_contrasts[contrast_name].volume

    def prepare_numpy_bundle(self, comment = ''):
        out_data = {'resolution': np.array(self.resolution), 'comment': comment}
        if self.affine is not None:
            out_data['affine'] = np.array(self.affine)

        # 'data' is the current contrast; the other contrasts are saved as 'data2', 'data3',
        # ..., mirroring the incremental-learning convention ('image', 'image2', ...).
        # Their names are kept in 'contrast_names' so that loading can restore them.
        if self.additional_contrasts:
            out_data['data'] = self._contrast_data_for_bundle(self.current_contrast)
            contrast_index = 2
            contrast_names = []
            for contrast_name in self.additional_contrasts:
                if contrast_name == self.current_contrast:
                    continue
                out_data[f'data{contrast_index}'] = self._contrast_data_for_bundle(contrast_name)
                contrast_names.append(contrast_name)
                contrast_index += 1
            if contrast_names:
                out_data['contrast_names'] = np.array(contrast_names)
        else:
            out_data['data'] = self.getDatasetAsNumpy()

        if self.has_time_dimension():
            # 4D masks with all the time frames
            allMasks = self._get_all_frames_masks()
        else:
            allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)
        for mask_name, mask in allMasks.items():
            out_data[f'mask_{mask_name}'] = mask
        return out_data
    
    def prepare_numpy_bundle_IL_3D(self, value_image, value_segm, meanDice, key, comment = ''):
        resolution = np.array(self.resolution)
        affine = np.array(self.affine)
        dice = np.array(meanDice)
        out_data = {'resolution': resolution, 'dice': dice, 'affine': affine.astype(np.float32), 'comment': comment, 'model': key}
        # value_image maps 'image', 'image2', ... to the corresponding contrast volume;
        # store them as 'data', 'data2', ... to keep the base-contrast key name backward compatible
        for contrast_key, contrast_value in value_image.items():
            data_key = 'data' if contrast_key == 'image' else f'data{contrast_key[len("image"):]}'
            out_data[data_key] = np.array(contrast_value).astype(np.float32)
        for sub_key in value_segm.keys():
            segm = np.array(value_segm[sub_key])
            out_data[f'mask_{sub_key}'] = segm.astype(np.uint8)
        return out_data

    @pyqtSlot(str)
    def uploadData(self, comment = ''):
        print('Uploading data')
        out_data = self.prepare_numpy_bundle(comment)
        self.model_provider.upload_data(out_data)
        self.setSplash(False, 2, 2, "Finished")

    @pyqtSlot(str)
    @snapshotSaver
    @separate_thread_decorator
    def loadMask(self, filename: str):
        dicom_ext = ['.dcm', '.ima']
        nii_ext = ['.nii', '.gz']
        npy_ext = ['.npy']
        npz_ext = ['.npz']
        path = os.path.abspath(filename)
        _, ext = os.path.splitext(path)

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
                else: # check if the folder contains dicoms
                    _, ext2 = os.path.splitext(new_path)
                    if ext2.lower() in dicom_ext:
                        containsDicom = True
                        if firstDicom is None:
                            firstDicom = new_path
                    elif ext2.lower() in nii_ext:
                        nii_list.append(new_path)

            if containsDicom and containsDirs:
                msgBox = QMessageBox()
                msgBox.setText('Folder contains both dicom files and subfolders.\nWhat do you want to do?')
                buttonDicom = msgBox.addButton('Load files as one ROI', QMessageBox.YesRole)
                buttonDir = msgBox.addButton('Load subfolders as multiple ROIs', QMessageBox.NoRole)
                msgBox.exec()
                if msgBox.clickedButton() == buttonDicom:
                    containsDirs = False
                else:
                    containsDicom = False

            if containsDicom:
                path = new_path # "fake" the loading of the first image
                _, ext = os.path.splitext(path)
            elif containsDirs:
                ext = 'multidicom' # "fake" extension to load a directory

        basename = os.path.basename(path)
        is3D = False

        self.setSplash(True, 0, 2, "Loading mask")

        def fail(text):
            self.setSplash(False, 0, 2, "Loading mask")
            self.alert(text, "Error")

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
            self.masksToRois({name: mask}, int(self.curImage)) # this is OK for 2D and 3D

        def align_masks(medical_volume):
            # check if 1) we have dicom headers to align the dataset and 2) the datasets are not already aligned
            if (self.affine is not None and
                    (not np.all(np.isclose(self.affine, medical_volume.affine, rtol=1e-3)) or
                     not np.all(medical_volume.shape == self.medical_volume.shape))):
                print("Aligning masks")
                self.setSplash(True, 1, 3, "Performing alignment")

                realigned_volume = realign_medical_volume(medical_volume, self.medical_volume, interpolation_order=0)

                mask = realigned_volume.volume
            else:
                # we cannot align the datasets
                print("Skipping alignment")
                mask = medical_volume.volume
            return mask

        def load_accumulated_mask(names, accumulated_mask):
            accumulated_mask = accumulated_mask.astype(np.uint16)
            if names is None:
                # load data without legend
                mask_values = np.unique(accumulated_mask)
                for index in mask_values:
                    if index == 0:
                        continue
                    print("Loading mask", index)
                    mask = np.zeros_like(accumulated_mask)
                    mask[accumulated_mask == index] = 1
                    load_mask_validate(str(index), mask)
                return

            for index, name in names.items():
                print("Loading mask", name, "with index", index)
                mask = np.zeros_like(accumulated_mask)
                mask[accumulated_mask == int(index)] = 1
                load_mask_validate(name, mask)

        def read_names_from_legend(legend_file):
            name_dict = {}
            with open(legend_file, newline='') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)
                for row in reader:
                    name_dict[row[0]] = row[1]
                    print(row[0], row[1])
            return name_dict


        ext = ext.lower()

        if ext in npy_ext:
            mask = np.load(path)
            name = basename
            self.setSplash(True, 1, 2, "Importing masks")
            load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return
        if ext in npz_ext:
            mask_dict = np.load(path)
            n_masks = len(mask_dict)
            cur_mask = 0
            for name, mask in mask_dict.items():
                self.setSplash(True, cur_mask, n_masks, "Importing masks")
                load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return
        elif ext in nii_ext:
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False, sort=GlobalConfig['DICOM_SORT'])
            name, _ = os.path.splitext(os.path.basename(path))

            mask = align_masks(mask_medical_volume)

            self.setSplash(True, 2, 3, "Importing masks")
            if mask.max() > 1: # dataset with multiple labels
                # try loading the legend
                legend_name = path + '.csv'
                try:
                    names = read_names_from_legend(legend_name)
                except FileNotFoundError:
                    self.alert(f'Legend file not found. Loading mask without legend.', 'Warning')
                    names = None
                load_accumulated_mask(names, mask)
            else:
                load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return
        elif ext in dicom_ext:
            # load dicom masks
            path = os.path.dirname(path)
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False, sort=GlobalConfig['DICOM_SORT'])
            name = os.path.basename(path)

            mask = align_masks(mask_medical_volume)
            self.setSplash(True, 2, 3, "Importing masks")
            if mask.max() > 1: # dataset with multiple labels
                # try loading the legend
                legend_name = os.path.join(path, 'legend.csv')
                try:
                    names = read_names_from_legend(legend_name)
                except FileNotFoundError:
                    self.alert(f'Legend file not found. Loading mask without legend.', 'Warning')
                    names = None
                load_accumulated_mask(names, mask)
            else:
                load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
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
                    mask_medical_volume, *_ = dosma_volume_from_path(data_path, reorient_data=False, sort=GlobalConfig['DICOM_SORT'])
                except:
                    continue
                dataset = mask_medical_volume.volume
                dataset[dataset > 0] = 1
                dataset[dataset < 1] = 0
                name, _ = os.path.splitext(os.path.basename(data_path))
                if accumulated_mask is None:
                    accumulated_mask = mask_medical_volume
                else:
                    try:
                        accumulated_mask.volume += dataset*current_mask_number
                    except:
                        print('Incompatible mask')
                        continue
                names.append(name)
                current_mask_number += 1
            if len(names) == 0:
                self.alert('No available mask found!')
                return

            aligned_masks = align_masks(accumulated_mask).astype(np.uint8)

            self.setSplash(True, 2, 3, "Importing masks")
            load_accumulated_mask(names, aligned_masks)
            self.setSplash(False, 0, 0, "")
            return

    @pyqtSlot(str)
    def save_data_as_reoriented_nifti(self, path):
        self.setSplash(True, 1, 3, "Saving data")
        reoriented_volume = reorient_data_ui(self.medical_volume, self.fig.canvas, inplace=False)
        nifti_writer = NiftiWriter()
        nifti_name = os.path.abspath(path)
        nifti_writer.save(reoriented_volume, nifti_name)
        self.setSplash(False, 0, 0, "")

    def _reorient_volume(self, volume, orientation):
        if orientation == 'Invert Slices':
            current_orientation = volume.orientation
            slc_orientation = current_orientation[2]
            new_slc_orientation = slc_orientation[1] + slc_orientation[0]
            return volume.reformat((current_orientation[0], current_orientation[1], new_slc_orientation))
        else:
            reoriented_volume = volume.reformat(get_nifti_orientation(orientation))
            reoriented_volume._headers = None
            return reoriented_volume

    @pyqtSlot(str)
    def reorient_data(self, orientation):
        print(orientation)
        if self.medical_volume is None:
            return
        medical_volume = self.medical_volume
        old_additional_contrasts = self.additional_contrasts
        old_additional_contrast_frames = self.additional_contrast_frames
        old_time_frames = self.time_frames
        self.resetInternalState()
        self.resetInterface()
        new_medical_volume = self._reorient_volume(medical_volume, orientation)
        self.load_dosma_volume(new_medical_volume)
        if len(old_time_frames) > 1:
            # reorient every time frame and make the first one the active volume
            self.time_frames = [self._reorient_volume(volume, orientation) for volume in old_time_frames]
            self.current_timepoint = 0
            self.medical_volume = self.time_frames[0]
            self.imList = ImListProxy(self.medical_volume)
            self.dicomHeaderList = None
        self.additional_contrasts[ToolboxWindow.BASE_CONTRAST_LABEL] = self.medical_volume
        for name, volume in old_additional_contrasts.items():
            if name == ToolboxWindow.BASE_CONTRAST_LABEL:
                continue
            if name in old_additional_contrast_frames:
                new_contrast_frames = [self._reorient_volume(frame, orientation)
                                       for frame in old_additional_contrast_frames[name]]
                self.additional_contrast_frames[name] = new_contrast_frames
                self.additional_contrasts[name] = new_contrast_frames[self.current_timepoint]
            else:
                self.additional_contrasts[name] = self._reorient_volume(volume, orientation)
            self.toolbox_window.add_contrast_to_combo(name)
        self._setup_timepoint_managers()
        # self.loadROIPickle()
        self.updateRoiList()
        self.override_class = None
        self.update_all_classifications()
        self.toolbox_window.set_exports_enabled(numpy= True,
                                                dicom= (self.dicomHeaderList is not None),
                                                nifti= (self.affine is not None)
                                                )
        self.axes.set_xlim(auto=True)
        self.axes.set_ylim(auto=True)
        self.displayImage(0)
        self.axes.set_xlim(auto=False)
        self.axes.set_ylim(auto=False)

    @pyqtSlot(str)
    def load_additional_contrast(self, filename):
        while True:
            accept, values = GenericInputDialog.show_dialog('Additional Contrast', [
                GenericInputDialog.TextLineInput('Name for additional contrast')
            ], self.fig.canvas)
            if not accept:
                return
            name = values[0]
            if self.toolbox_window.find_contrast_in_combo(name) >= 0:
                self.alert('Contrast name already in use!')
            else:
                break # exit loop if name is acceptable

        self._load_additional_contrast_data(filename, name)

    @separate_thread_decorator
    def _load_additional_contrast_data(self, filename, name):
        self.setSplash(True, 0, 1, "Loading additional contrast...")

        _, ext = os.path.splitext(filename)
        if ext.lower() == '.npz':
            try:
                bundle = np.load(filename, allow_pickle=False)
            except Exception as e:
                print(e, file=sys.stderr)
                self.alert("Error loading dataset. See the log for details", "Error")
                self.setSplash(False)
                return

            if 'data' in bundle:
                data = bundle['data']
            elif 'image' in bundle:
                data = bundle['image']
            else:
                self.alert('No data in bundle!', 'Error')
                self.setSplash(False)
                return

            if 'comment' in bundle:
                self.alert('Loading bundle with comment:\n' + str(bundle['comment']), 'Info')

            if data.ndim > 4:
                data = main_thread_dialog_runner.run(lambda: reduce_array_dimensions(data, self.fig.canvas))
                if data is None:
                    self.setSplash(False)
                    return

            affine = None
            if 'affine' in bundle:
                affine = bundle['affine']
            elif 'resolution' in bundle:
                resolution = list(bundle['resolution'])
                if len(resolution) == 2:
                    resolution.append(1.0)
                affine = np.diag(resolution + [1])

            affine_valid = affine is not None
            additional_volume = MedicalVolume(data, affine if affine_valid else np.eye(4))
        else:
            try:
                # reorient_data=False: the additional contrast is realigned/resampled onto the
                # base dataset's grid below anyway, so asking the user for a NIfTI orientation
                # here would be pointless. dosma_volume_from_path may still show other dialogs
                # (e.g. choosing a dataset from a multi-frame DICOM file), so route the whole
                # call through the main thread, since this method runs in a worker thread.
                additional_volume, affine_valid, _, _, _ = main_thread_dialog_runner.run(
                    lambda: dosma_volume_from_path(filename, self.fig.canvas,
                                                    reorient_data=False,
                                                    sort=GlobalConfig['DICOM_SORT']))
            except Exception as e:
                print(e, file=sys.stderr)
                self.alert("Error loading dataset. See the log for details", "Error")
                self.setSplash(False)
                return

            if self.has_time_dimension() and additional_volume.volume.ndim == 3:
                # a dicom stack can be a time-resolved acquisition in disguise
                header_list = self._header_list(additional_volume)
                regrouped_volume = self._regroup_dicom_time_series(additional_volume, header_list)
                if regrouped_volume is not None:
                    additional_volume = regrouped_volume

        if additional_volume.volume.ndim > 4:
            reduced_data = main_thread_dialog_runner.run(
                lambda: reduce_array_dimensions(additional_volume.volume, self.fig.canvas))
            if reduced_data is None:
                self.setSplash(False)
                return
            additional_volume = MedicalVolume(reduced_data, additional_volume.affine)

        if additional_volume.volume.ndim > 3:
            # time-resolved additional contrast: only allowed if it matches the frames of the dataset
            if not self.has_time_dimension() or additional_volume.shape[3] != self.n_timepoints:
                self.alert('The time frames of the additional contrast do not match the loaded dataset', 'Error')
                self.setSplash(False)
                return
            contrast_frames = [additional_volume[..., t] for t in range(additional_volume.shape[3])]
            if affine_valid:
                contrast_frames = [realign_medical_volume(frame, self.medical_volume)
                                   for frame in contrast_frames]
            elif contrast_frames[0].shape != self.medical_volume.shape:
                self.alert('The additional contrast dataset is not compatible with the loaded dataset', 'Error')
                self.setSplash(False)
                return
            self.additional_contrast_frames[name] = contrast_frames
            self.additional_contrasts[name] = contrast_frames[self.current_timepoint]
        else:
            if affine_valid:
                additional_volume = realign_medical_volume(additional_volume, self.medical_volume)
            elif additional_volume.shape != self.medical_volume.shape:
                self.alert('The additional contrast dataset is not compatible with the loaded dataset', 'Error')
                self.setSplash(False)
                return
            self.additional_contrasts[name] = additional_volume

        self.toolbox_window.add_contrast_to_combo(name)
        self.setSplash(False, 1, 1, "Loading additional contrast...")


    @pyqtSlot(str)
    def delete_additional_contrast(self, contrast_name):
        if contrast_name == ToolboxWindow.BASE_CONTRAST_LABEL:
            return
        if contrast_name not in self.additional_contrasts:
            return
        del self.additional_contrasts[contrast_name]
        self.additional_contrast_frames.pop(contrast_name, None)
        self.toolbox_window.remove_contrast_combo(contrast_name)

    @pyqtSlot(str)
    def change_contrast(self, contrast_name):
        if contrast_name not in self.additional_contrasts:
            print("Unknown contrast", contrast_name)
            return
        medical_volume = self.additional_contrasts[contrast_name]
        self.imList = ImListProxy(medical_volume)
        self.current_contrast = contrast_name
        self.contrastWindow = None
        self.displayImage(int(self.curImage))
        self.resetContrast()
        if self.toolbox_window.is_3D_viewer_visible():
            self.emit_viewer3d_data()


    ########################################################################################
    ###
    ### Time-resolved (4D) dataset support
    ###
    ########################################################################################

    @property
    def n_timepoints(self):
        return len(self.time_frames) if self.time_frames else 1

    def has_time_dimension(self):
        return len(self.time_frames) > 1

    def _split_time_frames(self):
        """ If the currently loaded medical volume is 4D, split it into a list of 3D volumes
            (one per time frame) and make the first frame the active volume. No-op for 3D data. """
        self.time_frames = []
        self.current_timepoint = 0
        if self.medical_volume is None or self.medical_volume.volume.ndim < 4:
            return
        volume_4d = self.medical_volume
        n_frames = volume_4d.shape[3]
        if n_frames > 1:
            self.time_frames = [volume_4d[..., t] for t in range(n_frames)]
            self.medical_volume = self.time_frames[0]
        else:
            self.medical_volume = volume_4d[..., 0] # trivial fourth dimension: treat as 3D
        self.dicomHeaderList = None # per-slice headers of the 4D stack don't apply to a single frame
        self.imList = ImListProxy(self.medical_volume)

    @staticmethod
    def _dicom_time_value(header, default=None):
        """ Extract a time marker from a dicom header, for sorting the frames of a
            time-resolved acquisition. """
        for attr in ('TriggerTime', 'TemporalPositionIdentifier', 'FrameReferenceTime'):
            value = getattr(header, attr, None)
            if value is not None and value != '':
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        value = getattr(header, 'AcquisitionTime', None) # HHMMSS.ffffff string
        if value:
            try:
                time_string = str(value)
                return int(time_string[0:2]) * 3600 + int(time_string[2:4]) * 60 + float(time_string[4:] or 0)
            except (TypeError, ValueError):
                pass
        return default

    @staticmethod
    def _header_list(medical_volume):
        """ Per-slice pydicom headers of a MedicalVolume as a flat list, or None. """
        if medical_volume.headers() is None:
            return None
        header_obj = medical_volume.headers().squeeze()
        if header_obj.shape == ():
            return [header_obj.item()]
        return list(header_obj)

    def _regroup_dicom_time_series(self, medical_volume, headers):
        """ Inspect the dicom headers of a 3D stack for repeated slice locations. If the stack
            is a regular slices x frames grid (a time-resolved acquisition), return the
            rearranged 4D MedicalVolume (frames sorted by time marker), otherwise None. """
        if medical_volume is None or medical_volume.volume.ndim != 3:
            return None
        n_total = medical_volume.shape[2]
        if not headers or len(headers) != n_total or n_total < 2:
            return None

        positions = []
        for header in headers:
            try:
                position = tuple(round(float(x), 2) for x in header.ImagePositionPatient)
            except (AttributeError, TypeError, ValueError):
                try:
                    position = (round(float(header.SliceLocation), 2),)
                except (AttributeError, TypeError, ValueError):
                    return None # no spatial information: nothing to detect
            positions.append(position)

        unique_positions = list(dict.fromkeys(positions)) # keep the first-appearance (spatial) order
        n_slices = len(unique_positions)
        if n_slices == n_total:
            return None # every image has its own location: plain 3D dataset
        if n_total % n_slices != 0:
            print('Repeated slice locations, but not a regular slices x frames grid: loading as 3D')
            return None
        n_frames = n_total // n_slices

        slice_groups = OrderedDict((position, []) for position in unique_positions)
        for index, position in enumerate(positions):
            time_value = self._dicom_time_value(headers[index], default=index)
            slice_groups[position].append((time_value, index))
        if any(len(group) != n_frames for group in slice_groups.values()):
            print('Repeated slice locations, but not a regular slices x frames grid: loading as 3D')
            return None

        volume = medical_volume.volume
        data_4d = np.empty(volume.shape[:2] + (n_slices, n_frames), dtype=volume.dtype)
        first_frame_headers = []
        for z, group in enumerate(slice_groups.values()):
            group.sort(key=lambda entry: entry[0])
            for t, (_, index) in enumerate(group):
                data_4d[:, :, z, t] = volume[:, :, index]
            first_frame_headers.append(headers[group[0][1]])

        try:
            affine = to_RAS_affine(first_frame_headers)
        except Exception as e:
            print('Could not recalculate the affine of the time-resolved dataset:', e)
            affine = medical_volume.affine

        return MedicalVolume(data_4d, affine)

    def _detect_dicom_time_series(self):
        """ Detect whether the loaded DICOM stack is a time-resolved acquisition: multiple
            slices sharing the same spatial location, possibly carrying time markers
            (TriggerTime and similar). If so, and the user confirms, rearrange the stack into
            a 4D volume, which _split_time_frames then turns into separate time frames. """
        volume_4d = self._regroup_dicom_time_series(self.medical_volume, self.dicomHeaderList)
        if volume_4d is None:
            return
        n_slices = volume_4d.shape[2]
        n_frames = volume_4d.shape[3]

        answer = QMessageBox.question(None, 'Time-resolved dataset',
                                      'This dataset looks time-resolved '
                                      f'({n_slices} slice(s) × {n_frames} time frames).\n'
                                      'Load it as a time-resolved (4D) dataset?',
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return

        self.medical_volume = volume_4d
        self.affine = self.medical_volume.affine
        self.resolution = np.array(self.medical_volume.pixel_spacing)
        self.resolution_valid = True
        self.dicomHeaderList = None # the per-slice headers of the mixed stack no longer apply
        self.imList = ImListProxy(self.medical_volume)
        print(f'Loading time-resolved dicom dataset: {n_slices} slice(s), {n_frames} time frames')

    def _timepoint_basename(self, timepoint):
        # timepoint 0 keeps the plain basename, so transform files of 3D datasets remain valid
        if timepoint == 0:
            return self.basename
        return (self.basename if self.basename else '') + '_t{}'.format(timepoint)

    def _setup_timepoint_managers(self):
        """ Create the per-timepoint ROI and registration managers and point the active ones
            to the current frame. For 3D datasets this creates a single manager pair. """
        mask_size = self.imList[0].shape
        self.roiManagers = {t: ROIManager(mask_size) for t in range(self.n_timepoints)}
        self.roiManager = self.roiManagers[self.current_timepoint]
        self.registrationManagers = {}
        for t in range(self.n_timepoints):
            if self.has_time_dimension():
                image_list = ImListProxy(self.time_frames[t])
            else:
                image_list = self.imList
            registration_manager = RegistrationManager(image_list,
                                                       None,
                                                       os.getcwd(),
                                                       GlobalConfig['TEMP_DIR'])
            registration_manager.set_standard_transforms_name(self.basepath, self._timepoint_basename(t))
            self.registrationManagers[t] = registration_manager
        self.registrationManager = self.registrationManagers[self.current_timepoint]
        self.toolbox_window.set_timepoints(self.n_timepoints)

    @pyqtSlot(int)
    def change_timepoint(self, timepoint):
        if not self.has_time_dimension():
            return
        timepoint = int(max(0, min(timepoint, len(self.time_frames) - 1)))
        if timepoint == self.current_timepoint:
            return
        self.current_timepoint = timepoint
        self.medical_volume = self.time_frames[timepoint]
        self.additional_contrasts[ToolboxWindow.BASE_CONTRAST_LABEL] = self.medical_volume
        # keep time-resolved additional contrasts in sync with the current frame
        for contrast_name, contrast_frames in self.additional_contrast_frames.items():
            self.additional_contrasts[contrast_name] = contrast_frames[timepoint]
        self.roiManager = self.roiManagers[timepoint]
        self.registrationManager = self.registrationManagers[timepoint]
        display_volume = self.additional_contrasts.get(self.current_contrast, self.medical_volume)
        self.imList = ImListProxy(display_volume)
        self.activeMask = None
        self.otherMask = None
        self.displayImage(int(self.curImage)) # also refreshes roi list, masks and contour painters
        self.redraw()
        if self.toolbox_window.is_3D_viewer_visible():
            self.emit_viewer3d_data()

    def next_timepoint(self):
        if self.has_time_dimension():
            # drive the toolbox slider, which propagates back to change_timepoint
            self.toolbox_window.set_current_timepoint(self.current_timepoint + 1)

    def previous_timepoint(self):
        if self.has_time_dimension():
            self.toolbox_window.set_current_timepoint(self.current_timepoint - 1)

    def _selected_roi_names(self, all_rois):
        """ ROI names a time operation should act on. With all_rois, take the union of the ROI
            names across every time frame: a ROI may only be segmented in frames other than
            the current one. """
        if not self.roiManager:
            return []
        if all_rois:
            roi_names = []
            for manager in self.roiManagers.values():
                for roi_name in manager.get_roi_names():
                    if roi_name not in roi_names:
                        roi_names.append(roi_name)
            return roi_names
        current_roi_name = self.getCurrentROIName()
        return [current_roi_name] if current_roi_name else []

    def _get_time_anchors(self, roi_name, slice_number):
        """ Return a dict timepoint -> mask with the nonempty masks of a ROI at a fixed slice
            across all the time frames. """
        anchors = {}
        for t in range(self.n_timepoints):
            mask = self.roiManagers[t].get_mask(roi_name, slice_number)
            if mask is not None and np.any(mask):
                anchors[t] = mask
        return anchors

    def _interpolate_mask_pair(self, mask_1, index_1, mask_2, index_2, target_index):
        """ Linear spline-based interpolation between two masks at generic positions
            index_1 < target_index < index_2. Returns None if interpolation is impossible. """
        spline_list_1 = mask_to_trivial_splines(mask_1, spacing=4)
        spline_list_2 = mask_to_trivial_splines(mask_2, spacing=4)
        if len(spline_list_1) != len(spline_list_2):
            self.alert('Different number of subrois in neighboring time frames')
            return None

        splines_list = masks_splines_to_splines_masks([spline_list_1, spline_list_2])
        out_mask = np.zeros(self.image.shape, dtype=np.uint8)
        for subroi_spline in splines_list:
            out_spline = SplineInterpROIClass()
            spline_1 = subroi_spline[0]
            spline_2 = subroi_spline[1]
            for knot_1, knot_2 in zip(spline_1.knots, spline_2.knots):
                f_x = interp1d([index_1, index_2], [knot_1[0], knot_2[0]], kind='linear')
                f_y = interp1d([index_1, index_2], [knot_1[1], knot_2[1]], kind='linear')
                out_spline.addKnot((f_x(target_index), f_y(target_index)))
            out_mask += out_spline.toMask(self.image.shape)
            out_mask = (out_mask > 0).astype(np.uint8)
            out_mask = binary_dilation(out_mask)
        return out_mask.astype(np.uint8)

    def _time_interpolate_mask(self, roi_name, slice_number, target_timepoint, anchors=None):
        """ Calculate the mask of a ROI at a fixed slice for one time frame from the frames
            where it is already segmented: linear interpolation between the two nearest
            anchor frames, or a copy of the nearest anchor if all the anchors lie on one side.
            Returns None if there is nothing to interpolate from. """
        if anchors is None:
            anchors = self._get_time_anchors(roi_name, slice_number)
        anchors = {t: mask for t, mask in anchors.items() if t != target_timepoint}
        if not anchors:
            return None
        timepoints_before = [t for t in anchors if t < target_timepoint]
        timepoints_after = [t for t in anchors if t > target_timepoint]
        if timepoints_before and timepoints_after:
            t_1 = max(timepoints_before)
            t_2 = min(timepoints_after)
            return self._interpolate_mask_pair(anchors[t_1], t_1, anchors[t_2], t_2, target_timepoint)
        nearest_timepoint = max(timepoints_before) if timepoints_before else min(timepoints_after)
        return anchors[nearest_timepoint].copy()

    def _sam_time_propagate(self, all_rois, inplace=True):
        """ Propagate the masks of the current slice through the whole time series with SAM2,
            treating the time frames at the fixed slice as a video and using every frame that
            already has a mask as an anchor. Mirrors samPropagateBlock, but along time.

            inplace=True: write the propagated masks into every non-anchor frame (the anchor
            frames -- the user's own masks -- are left untouched).
            inplace=False: return dict[roi_name -> dict[timepoint -> mask]] without touching
            the roiManagers. """
        if not self.has_time_dimension():
            return None
        slice_number = int(self.curImage)
        roi_names = self._selected_roi_names(all_rois)
        if not roi_names:
            return None

        masks_by_roi = {}
        time_bounds = {}
        for roi_name in roi_names:
            anchors = self._get_time_anchors(roi_name, slice_number)
            if not anchors:
                continue
            masks_by_roi[roi_name] = anchors
            time_bounds[roi_name] = (0, self.n_timepoints - 1)

        if not masks_by_roi:
            self.alert('No ROI is segmented on the current slice in any time frame')
            return None

        def progress_callback(current, maximum):
            self.setSplash(True, current, maximum, "SAM time propagation")

        time_stack = np.stack([frame.volume[:, :, slice_number].astype(np.float32)
                               for frame in self.time_frames])

        try:
            result = sam_api.SAM_propagate(
                time_stack, masks_by_roi, self.get_sam(),
                z_bounds=time_bounds,
                prompt_kind='mask', refine_mask_prompt=False,
                progress_callback=progress_callback)
        except Exception as e:
            print("Error in SAM time propagation:", e)
            self.alert("Error in SAM time propagation: " + str(e))
            self.setSplash(False)
            return None

        self.setSplash(False)

        if inplace:
            for roi_name, propagated in result.items():
                anchor_timepoints = set(masks_by_roi[roi_name])
                for t, mask in propagated.items():
                    if t in anchor_timepoints:
                        continue
                    if mask is None:
                        mask = np.zeros(self.image.shape, dtype=np.uint8)
                    self.roiManagers[t].set_mask(roi_name, slice_number, np.asarray(mask, dtype=np.uint8))
            return None
        return result

    @pyqtSlot(int, bool)
    @timeSnapshotSaver
    def time_copy(self, direction, all_rois):
        """ Copy the masks (all slices) of the current frame's ROI(s) to the adjacent time
            frame and move there. """
        if not self.has_time_dimension():
            return
        source_timepoint = self.current_timepoint
        target_timepoint = source_timepoint + direction
        if not (0 <= target_timepoint < self.n_timepoints):
            return
        roi_names = self._selected_roi_names(all_rois)
        if not roi_names:
            return
        source_manager = self.roiManagers[source_timepoint]
        target_manager = self.roiManagers[target_timepoint]
        copied = False
        for roi_name in roi_names:
            for key_tuple, mask in source_manager.all_masks(roi_name=roi_name):
                if mask is not None and np.any(mask):
                    target_manager.set_mask(key_tuple[0], key_tuple[1], mask.copy())
                    copied = True
        if not copied:
            self.alert('No masks to copy in the current frame')
            return
        self.toolbox_window.set_current_timepoint(target_timepoint)

    @pyqtSlot(str, bool)
    @timeSnapshotSaver
    @separate_thread_decorator
    def time_interpolate(self, interpolation_method, all_rois):
        """ Calculate the mask of the current slice in the current frame from the time frames
            where it is already segmented. """
        if not self.has_time_dimension():
            return
        slice_number = int(self.curImage)

        if interpolation_method == ToolboxWindow.INTERPOLATE_MASK_SAM:
            result = self._sam_time_propagate(all_rois, inplace=False)
            if not result:
                return
            for roi_name, propagated in result.items():
                mask = propagated.get(self.current_timepoint)
                if mask is not None:
                    self.roiManager.set_mask(roi_name, slice_number, np.asarray(mask, dtype=np.uint8))
        else:
            roi_names = self._selected_roi_names(all_rois)
            interpolated_any = False
            for roi_name in roi_names:
                new_mask = self._time_interpolate_mask(roi_name, slice_number, self.current_timepoint)
                if new_mask is None or not np.any(new_mask):
                    continue
                self.roiManager.set_mask(roi_name, slice_number, new_mask)
                interpolated_any = True
            if not interpolated_any:
                self.alert('No ROI is segmented on the current slice in any other time frame')
                return

        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.reblit()

    @pyqtSlot(str, bool)
    @timeSnapshotSaver
    @separate_thread_decorator
    def time_interpolate_block(self, interpolation_method, all_rois):
        """ Calculate the mask of the current slice in every time frame from the frames where
            it is already segmented. The anchor frames themselves are left untouched. """
        if not self.has_time_dimension():
            return
        slice_number = int(self.curImage)

        if interpolation_method == ToolboxWindow.INTERPOLATE_MASK_SAM:
            self._sam_time_propagate(all_rois, inplace=True)
        else:
            roi_names = self._selected_roi_names(all_rois)
            interpolated_any = False
            n_steps = len(roi_names) * self.n_timepoints
            current_step = 0
            for roi_name in roi_names:
                anchors = self._get_time_anchors(roi_name, slice_number)
                if not anchors:
                    current_step += self.n_timepoints
                    continue
                for t in range(self.n_timepoints):
                    self.setSplash(True, current_step, n_steps, "Interpolating in time...")
                    current_step += 1
                    if t in anchors:
                        continue
                    new_mask = self._time_interpolate_mask(roi_name, slice_number, t, anchors=anchors)
                    if new_mask is None or not np.any(new_mask):
                        continue
                    self.roiManagers[t].set_mask(roi_name, slice_number, new_mask)
                    interpolated_any = True
            self.setSplash(False)
            if not interpolated_any and not roi_names:
                return

        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.reblit()


    ########################################################################################
    ###
    ### Deep learning functions
    ###
    ########################################################################################

    @pyqtSlot(str, str)
    def importModel(self, modelFile, modelName):
        self.setSplash(True, 0, 1, 'Importing model...')

        modelName = modelName.replace('_', '-').replace(',', '.')

        try:
            self.model_provider.import_model(modelFile, modelName)
        except AttributeError:
            self.alert('Model provider does not support import')
            self.setSplash(False, 0, 1, 'Importing model...')
            return
        except Exception as err:
            self.alert('Error while importing model. Probably invalid model', 'Error')
            self.setSplash(False, 0, 1, 'Importing model...')
            traceback.print_exc()
            return
        self.setSplash(True, 1, 1, 'Importing model...')
        self.alert('Model imported successfully', 'Info')
        self.setSplash(False, 1, 1, 'Importing model...')
        GlobalConfig['ENABLED_MODELS'].append(modelName)
        self.setAvailableClasses(self.model_provider.available_models())

    def setModelProvider(self, modelProvider):
        self.model_provider = modelProvider
        if GlobalConfig['USE_CLASSIFIER']:
            try:
                self.dl_classifier = modelProvider.load_model('Classifier', force_download=GlobalConfig['FORCE_MODEL_DOWNLOAD'])
            except:
                self.dl_classifier = None
        else:
            self.dl_classifier = None

    def setAvailableClasses(self, classList, filter_classes = False):
        original_classifications = self.classifications[:]
        try:
            classList.remove('Classifier')
        except ValueError: # Classifier doesn't exist. It doesn't matter
            pass

        new_class_list = []
        self.model_details = {}
        for c in classList:
            if self.model_provider is None:
                new_class_list.append(c)
            else:
                model_details = self.model_provider.model_details(c)
                self.model_details[c] = model_details
                # if filter_classes, only show explicitly enabled models
                if filter_classes and c not in GlobalConfig['ENABLED_MODELS']:
                    continue
                try:
                    variants = model_details['variants']
                except:
                    new_class_list.append(c)
                    continue
                for variant in variants:
                    if variant.strip() == '':
                        new_class_list.append(c)
                    else:
                        new_class_list.append(f'{c}, {variant}')

        torch.cuda.empty_cache()

        for i, classification in enumerate(original_classifications[:]):
            if classification not in new_class_list:
                original_classifications[i] = 'None'
        self.toolbox_window.set_available_classes(new_class_list, self.model_details)

        try:
            self.toolbox_window.set_class(original_classifications[int(self.curImage)])  # update the classification combo
        except IndexError:
            pass

    @pyqtSlot(str)
    @pyqtSlot(str)
    def changeClassification(self, newClass):
        try:
            self.classifications[int(self.curImage)] = newClass
        except IndexError:
            print("Trying to change classification to an unexisting image")

    @pyqtSlot(str)
    def changeAllClassifications(self, newClass):
        for i in range(len(self.classifications)):
            self.classifications[i] = newClass

    @pyqtSlot(int, int)
    @separate_thread_decorator
    def doSegmentationMultislice(self, min_slice, max_slice):
        if min_slice > max_slice: # invert order if one is bigger than the other
            min_slice, max_slice = max_slice, min_slice
        
        if self._is_current_model_3D():
            self.displayImage(min_slice)
            self.doSegmentation_3D(min_slice, max_slice)
            self.setSplash(True, 0, 3, "Loading model...")
            time.sleep(0.5)
            
        else:
            for slice_number in range(min_slice, max_slice+1):
                self.displayImage(slice_number)
                self.doSegmentation()
                self.setSplash(True, 0, 3, "Loading model...")
                time.sleep(0.5)
        self.setSplash(False, 0, 3, "End of SegmentationMultislice")

    def getSegmentedMasks(self, imIndex, setSplash=False, downloadModel=True):
        class_str = self.classifications[imIndex]
        if class_str == 'None':
            self.alert('Segmentation with "None" model is impossible!', 'Error')
            return

        if setSplash:
            self.setSplash(True, 0, 3, "Loading model...")

        segmenter, model_str = self.get_model_for_class(class_str, downloadModel, setSplash)

        if setSplash:
            self.setSplash(True, 1, 3, "Running segmentation...")

        image = self.imList[imIndex]
        subregion = None
        if self.toolbox_window.get_subregion_restriction():
            subregion = self.roiManager.get_autosegment_subregion(imIndex)
            image = image[subregion[0]:(subregion[0] + subregion[2]), subregion[1]:(subregion[1]+subregion[3])]

        inputData = {'image': image, 'resolution': self.resolution[0:2],
                     'split_laterality': GlobalConfig['SPLIT_LATERALITY'], 'classification': class_str}

        image_index = 2
        for contrast_name, contrast_volume in self.additional_contrasts.items():
            if contrast_name == self.current_contrast:
                continue
            other_image = contrast_volume.volume[:, :, imIndex].astype(np.float32)
            if subregion is not None:
                other_image = other_image[subregion[0]:(subregion[0] + subregion[2]), subregion[1]:(subregion[1]+subregion[3])]
            inputData[f'image{image_index}'] = other_image
            image_index += 1

        print("Segmenting image...")
        masks_out = segmenter(inputData)
        if self.toolbox_window.get_subregion_restriction():
            # reformat the masks
            new_masks_out = {}
            for mask_name, mask in masks_out.items():
                new_masks_out[mask_name] = np.zeros_like(self.imList[imIndex])
                new_masks_out[mask_name][subregion[0]:(subregion[0] + subregion[2]), subregion[1]:(subregion[1]+subregion[3])] = mask
            masks_out = new_masks_out
        return masks_out

    def getSegmentedMasks_3D(self, imIndex, setSplash=False, downloadModel=True):
        print("Set splash:", setSplash)

        class_str = self.classifications[int(imIndex[0])]

        if class_str == 'None':
            self.alert('Segmentation with "None" model is impossible!', 'Error')
            return

        model_str = class_str.split(',')[0].strip()  # get the base model string in case of multiple variants.
        # variants are identified by "Model, Variant"

        if setSplash:
            print("Setting splash")
            self.setSplash(True, 0, 3, "Loading model...")

        segmenter, model_str = self.get_model_for_class(class_str, downloadModel, setSplash)

        if setSplash:
            self.setSplash(True, 1, 3, "Running segmentation...")

        current_volume = self.additional_contrasts.get(self.current_contrast, self.medical_volume)
        image = current_volume[:,:,imIndex[0]:imIndex[-1]+1]
        image = ensure_compatible_orientation(image, segmenter.get_metadata())
        print("Resolution", self.resolution)
        print("Affine", self.affine)
        if self.affine is not None:
            affine = self.affine
        else:
            affine = np.diag([*self.resolution, 1.0])
        #affine = self.affine or np.diag([1.0, 1.0, 1.0, 1.0])
        inputData = {'image': image.volume.astype(np.float32), 'affine': affine, 'resolution': self.resolution,
                    'split_laterality': False, 'classification': class_str}

        image_index = 2
        for contrast_name, contrast_volume in self.additional_contrasts.items():
            if contrast_name == self.current_contrast:
                continue
            other_image = contrast_volume[:,:,imIndex[0]:imIndex[-1]+1]
            other_image = ensure_compatible_orientation(other_image, segmenter.get_metadata())
            inputData[f'image{image_index}'] = other_image.volume.astype(np.float32)
            image_index += 1

        print("Segmenting image...")
        masks_out = segmenter(inputData)
        for key, mask in masks_out.items():
            print("Total mask voxels", key, np.sum(mask))
        torch.cuda.empty_cache()
        return masks_out
    
    @pyqtSlot()
    @snapshotSaver
    def doSegmentation(self):
        # run the segmentation
        imIndex = int(self.curImage)

        print("2D Segmentation")

        t = time.time()
        masks_out=self.getSegmentedMasks(imIndex, True, True)
        if masks_out is None:
            self.setSplash(False, 0, 3, "Loading model...")
            return
        self.setSplash(True, 2, 3, "Converting masks...")
        print("Done")
        self.masksToRois(masks_out, imIndex)
        self.activeMask = None
        self.otherMask = None
        print("Segmentation/import time:", time.time() - t)
        self.setSplash(False, 3, 3)
        time.sleep(0.1)
        self.redraw()

    @pyqtSlot(int, int)
    @snapshotSaver
    def doSegmentation_3D(self, min_slice, max_slice):
        # run the segmentation
        image=self.medical_volume
        imIndex = range(min_slice,max_slice+1)

        print("3D segmentation")

        t = time.time()
        masks_out=self.getSegmentedMasks_3D(imIndex, True, True)

        if masks_out is None:
            self.setSplash(False, 0, 3, "Loading model...")
            return
        
        self.setSplash(True, 2, 3, "Converting masks...")
        print("Done")
        self.masksToRois(masks_out, image[:,:,imIndex[0]:imIndex[-1]+1])
        self.activeMask = None
        self.otherMask = None # TODO: Is it correct?
        print("Segmentation/import time:", time.time() - t)
        self.setSplash(False, 3, 3)
        time.sleep(0.1)
        self.redraw()

    #@pyqtSlot()
    @separate_thread_decorator # this might crash tensorflow. Remove in case of problems
    @pyqtSlot()
    def incrementalLearnStandalone(self):
        model, model_str = self.get_model_for_class(self.classifications[int(self.curImage)])
        if not model.can_incremental_learn():
            self.alert("This model cannot perform incremental learning")
            return

        if self._is_current_model_3D():
            self.setSplash(True, 0, 4, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)
        self.setSplash(True, 1, 4, "Incremental learning...")

        # perform incremental learning
        if self._is_current_model_3D():
            self.update_3D_incrLearn_objects()
            self.save_3D_bundle_for_IL(dataForTraining, segForTraining, meanDiceScore) # save current bundle if called
            self.incrementalLearn_3D(self.incrLearnDataTrain, self.incrLearnSegTrain, self.incrementalLearningAffine, self.incrLearnMeanDice, True)
        else:
            self.incrementalLearn(dataForTraining, segForTraining, meanDiceScore, True)
        self.setSplash(False, 3, 4, "Saving file...")

    def get_model_for_class(self, classification_name, download_model=True, set_splash=True):
        model_str = classification_name.split(',')[0].strip()  # get the base model string in case of multiple variants.
        # variants are identified by "Model, Variant"

        try:
            model = self.dl_segmenters[model_str]
        except KeyError:
            if download_model:
                if set_splash:
                    splashCallback = lambda cur_val, max_val: self.setSplash(True, cur_val, max_val,
                                                                                               'Downloading Model...')
                else:
                    splashCallback = None
                model = self.model_provider.load_model(model_str, splashCallback,
                                                       force_download=GlobalConfig['FORCE_MODEL_DOWNLOAD'])
                if model is None:
                    self.setSplash(False, 0, 3, "Loading model...")
                    self.alert(f"Error loading model {model_str}", 'Error')
                    return None, model_str
                self.dl_segmenters[classification_name] = model
            else:
                return None, model_str
        return model, model_str

    def incrementalLearn(self, dataForTraining, segForTraining, meanDiceScore, setSplash=False):
        performed = False
        for classification_name in dataForTraining:
            if classification_name == 'None': continue
            print(f'Performing incremental learning for {classification_name}')
            if len(dataForTraining[classification_name]) < GlobalConfig['IL_MIN_SLICES']:
                print(f"Not enough slices for {classification_name}")
                continue
            performed = True

            model, model_str = self.get_model_for_class(classification_name)

            if not model.can_incremental_learn():
                print("This model cannot perform incremental learning")
                return
            training_outputs = []

            # dataForTraining[classification_name] maps a slice index to a dict of
            # 'image', 'image2', ... -> slice. Only contrasts common to every slice
            # can be aligned into per-contrast lists.
            slice_indices = list(dataForTraining[classification_name].keys())
            common_contrast_keys = set(dataForTraining[classification_name][slice_indices[0]].keys())
            all_contrast_keys = set(common_contrast_keys)
            for imageIndex in slice_indices[1:]:
                keys = set(dataForTraining[classification_name][imageIndex].keys())
                common_contrast_keys &= keys
                all_contrast_keys |= keys
            if common_contrast_keys != all_contrast_keys:
                print(f"Warning: inconsistent contrasts across slices for {classification_name}; "
                      f"only using contrasts common to all slices: {sorted(common_contrast_keys)}")

            training_data_by_contrast = {contrast_key: [] for contrast_key in common_contrast_keys}
            for imageIndex in slice_indices:
                for contrast_key in common_contrast_keys:
                    training_data_by_contrast[contrast_key].append(dataForTraining[classification_name][imageIndex][contrast_key])
                training_outputs.append(segForTraining[classification_name][imageIndex])
                self.slicesUsedForTraining.add(imageIndex) # add the slice to the set of already used ones

            try:
                # todo: adapt bs and minTrainImages if needed
                training_payload = {'resolution': self.resolution[0:2], 'classification': classification_name}
                for contrast_key, contrast_list in training_data_by_contrast.items():
                    list_key = 'image_list' if contrast_key == 'image' else f'{contrast_key}_list'
                    training_payload[list_key] = contrast_list
                model.incremental_learn(training_payload,
                                        training_outputs, bs=5, minTrainImages=GlobalConfig['IL_MIN_SLICES'])
                model.reset_timestamp()
            except Exception as e:
                print("Error during incremental learning")
                traceback.print_exc()

            # Uploading new model

            # Only upload delta, to reduce model size -> only activate if rest of federated learning
            # working properly
            # all weights lower than threshold will be set to 0 for model compression
            # threshold = 0.0001
            # model = model.calc_delta(orig_model, threshold=threshold)
            if setSplash:
                self.setSplash(True, 2, 4, "Sending the improved model to server...")

            st = time.time()
            if meanDiceScore is None:
                meanDiceScore = -1.0
            self.model_provider.upload_model(model_str, model, meanDiceScore)
            print(f"took {time.time() - st:.2f}s")
        if not performed:
            self.alert("Not enough images for incremental learning")

    def incrementalLearn_3D(self, dataForTraining, segForTraining, incrementalLearningAffine, incrLearnDiceScore, setSplash=False):
        performed = False
        print("Classifications in Training data:", list(dataForTraining.keys()))
        for classification_name in dataForTraining:
            if classification_name == 'None': continue

            model = None

            # do not load the model yet, in case it cannot do incremental learning
            can_learn = get_model_detail(self.model_details, classification_name, 'can_incremental_learn', None)

            if can_learn is None:
                # incremental learning capability is not in the details. Load the model now
                model, model_str = self.get_model_for_class(classification_name)
                if model is None:
                    can_learn = False
                else:
                    can_learn = model.can_incremental_learn()
            if not can_learn:
                print("This model cannot perform incremental learning")
                continue

            print(f'Performing incremental learning for {classification_name}')
            if len(dataForTraining[classification_name]) < GlobalConfig['IL_3D_MIN_IMAGES']:
                print(f"Not enough images for {classification_name}")
                continue

            performed = True # if we reached here, then we can perform IL
            # mean dice scores
            diceScores = []

            for imageIndex in incrLearnDiceScore[classification_name]:
                diceScores.append(incrLearnDiceScore[classification_name][imageIndex])

            diceScores = np.array(diceScores)
            # print("diceScores array: ", diceScores)
            meanDiceScore = np.average(diceScores) #, weights=diceScores.size)
            print("mean dice score: ", meanDiceScore)


            training_outputs = {}
            training_affine = []

            # dataForTraining[classification_name] maps a session index to a dict of
            # 'image', 'image2', ... -> volume. Only contrasts common to every saved
            # session can be aligned into per-contrast lists; sessions saved before a
            # given additional contrast was loaded (or without it) are missing that key.
            session_indices = sorted(dataForTraining[classification_name].keys())
            sessions = [dataForTraining[classification_name][i] for i in session_indices]
            common_contrast_keys = set(sessions[0].keys())
            all_contrast_keys = set(sessions[0].keys())
            for session in sessions[1:]:
                common_contrast_keys &= set(session.keys())
                all_contrast_keys |= set(session.keys())
            if common_contrast_keys != all_contrast_keys:
                print(f"Warning: inconsistent contrasts across saved incremental-learning sessions for "
                      f"{classification_name}; only using contrasts common to all sessions: {sorted(common_contrast_keys)}")

            training_data_by_contrast = {
                contrast_key: [session[contrast_key].astype(np.float32) for session in sessions]
                for contrast_key in common_contrast_keys
            }

            for imageIndex in incrementalLearningAffine[classification_name]:
                training_affine.append(incrementalLearningAffine[classification_name][imageIndex].astype(np.float32))

            for imageIndex in segForTraining[classification_name]:
                for sub_key in segForTraining[classification_name][0].keys():
                    training_outputs[imageIndex] = {sub_key: segForTraining[classification_name][imageIndex][sub_key].astype(np.uint8)}

            if model is None:
                model, model_str = self.get_model_for_class(classification_name) # if we didn't need to load the model before, do it now

            try:
                gc.collect()
                torch.cuda.empty_cache()
                # todo: adapt bs and minTrainImages if needed
                training_payload = {'affine': training_affine, 'resolution': self.resolution, 'classification': classification_name}
                for contrast_key, contrast_list in training_data_by_contrast.items():
                    list_key = 'image_list' if contrast_key == 'image' else f'{contrast_key}_list'
                    training_payload[list_key] = contrast_list
                model.incremental_learn(training_payload,
                                        training_outputs, bs=1, minTrainImages=GlobalConfig['IL_3D_MIN_IMAGES'])
                gc.collect()
                torch.cuda.empty_cache()
                model.reset_timestamp()
            except Exception as e:
                print("Error during incremental learning")
                traceback.print_exc()

            # Uploading new model

            # Only upload delta, to reduce model size -> only activate if rest of federated learning
            # working properly
            # all weights lower than threshold will be set to 0 for model compression
            # threshold = 0.0001
            # model = model.calc_delta(orig_model, threshold=threshold)
            if setSplash:
                self.setSplash(True, 2, 4, "Sending the improved model to server...")

            st = time.time()

            self.model_provider.upload_model(model_str, model, meanDiceScore)
            print(f"took {time.time() - st:.2f}s")
            # Cleanup training data
            directory = os.path.join(GlobalConfig['NUMPY_FILE_3D'], classification_name)
            for file in os.listdir(directory):
                if re.fullmatch(r'temp_[0-9]+\.npz', file):
                    print('Deleting', file)
                    os.unlink(os.path.join(directory, file))


        if not performed:
            alert_text = f'Not enough data points for 3D incremental learning. Required: {GlobalConfig["IL_3D_MIN_IMAGES"]}.\nAvailable datasets:'
            for class_name in dataForTraining.keys():
                alert_text += f'\n{class_name} -> {len(dataForTraining[class_name])}'
            alert_text += '\nThe current data point was saved for future learning'
            self.alert(alert_text)
