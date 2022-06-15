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
import matplotlib

from ..utils.dicomUtils.misc import realign_medical_volume, dosma_volume_from_path
from . import GenericInputDialog

matplotlib.use("Qt5Agg")

import os, time, math, sys

from ..config import GlobalConfig, load_config
load_config()

from .ToolboxWindow import ToolboxWindow
from .pyDicomView import ImageShow
from ..utils.mask_utils import save_npy_masks, save_npz_masks, save_dicom_masks, save_nifti_masks, \
    save_single_dicom_dataset, save_single_nifti
from dafne_dl.misc import calc_dice_score
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import shutil
from datetime import datetime
from ..utils.ROIManager import ROIManager

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from ..utils import compressed_pickle as pickle
import os.path
from collections import deque
import functools
import csv

from ..utils.ThreadHelpers import separate_thread_decorator

from .BrushPatches import SquareBrush, PixelatedCircleBrush
from .ContourPainter import ContourPainter
import traceback

from dafne_dl.LocalModelProvider import LocalModelProvider
from dafne_dl.RemoteModelProvider import RemoteModelProvider

from ..utils.RegistrationManager import RegistrationManager

import requests

try:
    import SimpleITK as sitk # this requires simpleelastix! It is NOT available through PIP
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


class MuscleSegmentation(ImageShow, QObject):

    undo_possible = pyqtSignal(bool)
    redo_possible = pyqtSignal(bool)
    splash_signal = pyqtSignal(bool, int, int, str)
    reblit_signal = pyqtSignal()
    redraw_signal = pyqtSignal()
    reduce_brush_size = pyqtSignal()
    increase_brush_size = pyqtSignal()
    alert_signal = pyqtSignal(str)
    undo_signal = pyqtSignal()
    redo_signal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        self.suppressRedraw = False
        ImageShow.__init__(self, *args, **kwargs)
        QObject.__init__(self)
        self.fig.canvas.mpl_connect('close_event', self.closeCB)
        # self.instructions = "Shift+click: add point, Shift+dblclick: optimize/simplify, Ctrl+click: remove point, Ctrl+dblclick: delete ROI, n: propagate fw, b: propagate back"
        self.setupToolbar()

        self.roiManager = None

        self.wacom = False

        self.saveDicom = False

        self.model_provider = None
        self.dl_classifier = None
        self.dl_segmenters = {}

        # self.fig.canvas.setCursor(Qt.BlankCursor)
        self.app = None

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

        # disable keymapping from matplotlib - avoid pan and zoom
        for key in list(plt.rcParams):
            if 'keymap' in key and 'zoom' not in key and 'pan' not in key:
                plt.rcParams[key] = []

    def get_app(self):
        if not self.app:
            self.app = QApplication.instance()
        return self.app


    def resizeCB(self, event):
        self.resetBlitBg()
        self.redraw()

    def resetBlitBg(self):
        self.blitBg = None

    def resetModelProvider(self):
        available_models = None
        if GlobalConfig['MODEL_PROVIDER'] == 'Local':
            model_provider = LocalModelProvider(GlobalConfig['MODEL_PATH'], GlobalConfig['TEMP_UPLOAD_DIR'])
            available_models = model_provider.available_models()
        else:
            model_provider = RemoteModelProvider(GlobalConfig['MODEL_PATH'], GlobalConfig['SERVER_URL'], GlobalConfig['API_KEY'], GlobalConfig['TEMP_UPLOAD_DIR'])
            fallback = False
            try:
                available_models = model_provider.available_models()
            except PermissionError:
                self.alert("Error in using Remote Model. Please check your API key. Falling back to Local")
                fallback = True
            except requests.exceptions.ConnectionError:
                self.alert("Remote server unavailable. Falling back to Local")
                fallback = True
            else:
                if available_models is None:
                    self.alert("Error in using Remote Model Loading. Falling back to Local")
                    fallback = True

            if fallback:
                GlobalConfig['MODEL_PROVIDER'] = 'Local'
                model_provider = LocalModelProvider(GlobalConfig['MODEL_PATH'], GlobalConfig['TEMP_UPLOAD_DIR'])
                available_models = model_provider.available_models()

        self.setModelProvider(model_provider)

        print(available_models)
        self.setAvailableClasses(available_models)

    @pyqtSlot()
    def configChanged(self):
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

        self.maskImPlot = None
        self.maskOtherImPlot = None
        self.activeMask = None
        self.otherMask = None

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
        #load_config() # this was already loaded in dafne.py
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


    #############################################################################################
    ###
    ### Toolbar interaction
    ###
    ##############################################################################################

    def setupToolbar(self):

        if 'Elastix' in dir(sitk):
            showRegistrationGui = True
        else:
            print("Elastix is not available")
            showRegistrationGui = False

        self.toolbox_window = ToolboxWindow(self, activate_registration=showRegistrationGui, activate_radiomics= (radiomics is not None))
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

        self.toolbox_window.roi_import.connect(self.loadROIPickle)
        self.toolbox_window.roi_export.connect(self.saveROIPickle)
        self.toolbox_window.data_open.connect(self.loadDirectory)
        self.toolbox_window.masks_export.connect(self.saveResults)

        self.toolbox_window.roi_copy.connect(self.copyRoi)
        self.toolbox_window.roi_combine.connect(self.combineRoi)
        self.toolbox_window.roi_multi_combine.connect(self.combineMultiRoi)
        self.toolbox_window.roi_remove_overlap.connect(self.roiRemoveOverlap)

        self.toolbox_window.statistics_calc.connect(self.saveStats)
        self.toolbox_window.radiomics_calc.connect(self.saveRadiomics)

        self.toolbox_window.incremental_learn.connect(self.incrementalLearnStandalone)

        self.toolbox_window.mask_import.connect(self.loadMask)

        self.splash_signal.connect(self.toolbox_window.set_splash)
        self.interface_disabled = False
        self.splash_signal.connect(self.disableInterface)

        self.toolbox_window.mask_grow.connect(self.maskGrow)
        self.toolbox_window.mask_shrink.connect(self.maskShrink)

        self.toolbox_window.config_changed.connect(self.configChanged)
        self.toolbox_window.data_upload.connect(self.uploadData)

        self.toolbox_window.model_import.connect(self.importModel)

        self.reduce_brush_size.connect(self.toolbox_window.reduce_brush_size)
        self.increase_brush_size.connect(self.toolbox_window.increase_brush_size)
        self.toolbox_window.brush_changed.connect(self.updateBrush)

        self.alert_signal.connect(self.toolbox_window.alert)

        self.toolbox_window.reblit.connect(self.do_reblit)

    def setSplash(self, is_splash, current_value = 0, maximum_value = 1, text= ""):
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

    def alert(self, text):
        self.alert_signal.emit(text)

    #############################################################################################
    ###
    ### History
    ###
    #############################################################################################

    def saveSnapshot(self, save_head = False):
        #print("Saving snapshot")
        if self.roiManager is None:
            try:
                self.roiManager = ROIManager(self.imList[0].shape)
            except:
                return
        current_point = pickle.dumps(self.roiManager)
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
        self.clearAllROIs()
        if self.currentHistoryPoint == 0:
            #print("loading head")
            self.roiManager = pickle.loads(self.historyHead)
            self.historyHead = None
        else:
            #print("loading", self.currentHistoryPoint-1)
            self.roiManager = pickle.loads(self.history[self.currentHistoryPoint-1])

        self.updateRoiList()
        if self.roiManager.contains(roiName):
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
        #print(roi_name, subroi_index)
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
        self.reblit()

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
            if setSplash:
                self.setSplash(True, current_roi_index, len(roi_names), "Calculating maps...")
                current_roi_index += 1
            masklist = []
            for imageIndex in range(len(self.imList)):
                roi = np.zeros(imSize)
                if self.roiManager.contains(roiName, imageIndex):
                    roi = self.roiManager.get_mask(roiName, imageIndex)

                if roi.any():
                    slices_with_rois.add(imageIndex) # add the slice to the set if any voxel is nonzero
                    if imageIndex not in originalSegmentationMasks:
                        #print(imageIndex)
                        originalSegmentationMasks[imageIndex] = self.getSegmentedMasks(imageIndex, False, True)

                masklist.append(roi)
                try:
                    originalSegmentation = originalSegmentationMasks[imageIndex][roiName]
                except:
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
                    dataForTraining[classification_name][imageIndex] = self.imList[imageIndex]
                    segForTraining[classification_name][imageIndex] = {}

                segForTraining[classification_name][imageIndex][roiName] = roi

            npMask = np.transpose(np.stack(masklist), [1, 2, 0])
            allMasks[roiName] = npMask

        # cleanup empty slices and slices that were already used for training
        for classification_name in dataForTraining:
            print('Slices available for', classification_name, ':', list(dataForTraining[classification_name].keys()))
            for imageIndex in list(dataForTraining[classification_name]): # get a list of keys to be able to delete from dict
                if imageIndex not in slices_with_rois or imageIndex in self.slicesUsedForTraining:
                    del dataForTraining[classification_name][imageIndex]
                    del segForTraining[classification_name][imageIndex]
            print('Slices after cleanup', list(dataForTraining[classification_name].keys()))


        diceScores = np.array(diceScores)
        n_voxels =np.array(n_voxels)
        #print(diceScores)
        if np.sum(n_voxels) == 0:
            average_dice = -1.0
        else:
            average_dice = np.average(diceScores, weights=n_voxels)
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


    #########################################################################################
    ###
    ### ROI modifications
    ###
    #########################################################################################

    @snapshotSaver
    def simplify(self):
        r = self.getCurrentROI()
        # self.setCurrentROI(r.getSimplifiedSpline(GlobalConfig['SIMPLIFIED_ROI_POINTS']))
        # self.setCurrentROI(r.getSimplifiedSpline(spacing=GlobalConfig['SIMPLIFIED_ROI_SPACING']))
        self.setCurrentROI(r.getSimplifiedSpline3())
        self.reblit()

    @snapshotSaver
    def optimize(self):
        print("Optimizing ROI")
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
                print("Optimizing existing knots")
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
        self.displayImage(self.imList[int(self.curImage)], self.cmap)
        self.redraw()
        self.setSplash(True, 1, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            self.simplify()
            self.setSplash(True, 2, 3)
            self.optimize()

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
                print("Optimizing existing knots")
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
        self.displayImage(self.imList[int(self.curImage)], self.cmap)
        self.redraw()

        self.setSplash(True, 2, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            self.simplify()
            self.setSplash(True, 3, 3)
            self.optimize()

        self.setSplash(False, 3, 3)

    ##############################################################################################################
    ###
    ### Displaying
    ###
    ###############################################################################################################


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

    def updateMasksFromROIs(self):
        roi_name = self.getCurrentROIName()
        mask_size = self.image.shape
        self.otherMask = np.zeros(mask_size, dtype=np.uint8)
        self.activeMask = np.zeros(mask_size, dtype=np.uint8)
        for key_tuple, mask in self.roiManager.all_masks(image_number=self.curImage):
            mask_name = key_tuple[0]
            if mask_name == roi_name:
                self.activeMask = mask.copy()
            else:
                self.otherMask = np.logical_or(self.otherMask, mask)

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
                                               vmin=0, vmax=1, zorder=100, aspect=self.resolution[1]/self.resolution[0])
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
            self.maskOtherImPlot = self.axes.imshow(other_mask, cmap=self.mask_layer_other_colormap,
                                                    alpha=relativeAlphaROI*GlobalConfig['MASK_LAYER_ALPHA'],
                                                    vmin=0, vmax=1, zorder=101, aspect=self.resolution[1]/self.resolution[0])
            try:
                self.axes.set_xlim(original_xlim)
                self.axes.set_ylim(original_ylim)
            except:
                pass
            self.maskOtherImPlot.set_animated(True)

        self.maskOtherImPlot.set_data(other_mask.astype(np.uint8))
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

    # convert a single slice to ROIs
    def maskToRois2D(self, name, mask, imIndex, refresh = True):
        if not self.roiManager: return
        self.roiManager.set_mask(name, imIndex, mask)
        if refresh:
            self.updateRoiList()
            self.redraw()

    # convert a 2D mask or a 3D dataset to rois
    def masksToRois(self, maskDict, imIndex):
        for name, mask in maskDict.items():
            if len(mask.shape) > 2: # multislice
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
        ImageShow.displayImage(self, im, cmap, redraw)
        self.updateRoiList()  # set the appropriate (sub)roi list for the current image
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
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

    def redraw(self):
        self.redraw_signal.emit()

    @pyqtSlot()
    def do_redraw(self):
        #print("Redrawing...")
        if self.suppressRedraw: return
        #print("Yes")
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
        self.fig.canvas.draw()
        self.suppressRedraw = True # avoid nested calls
        self.fig.canvas.flush_events()
        #plt.pause(0.00001)
        self.suppressRedraw = False
        self.blitBg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.blitXlim = self.axes.get_xlim()
        self.blitYlim = self.axes.get_ylim()
        self.refreshCB()
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


    def closeCB(self, event):
        self.toolbox_window.close()
        if not self.basepath: return
        if self.registrationManager:
            self.registrationManager.pickle_transforms()
        self.saveROIPickle()
        sys.exit(0)

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
            np.logical_or(self.activeMask, self.brush_patch.to_mask(self.activeMask.shape), out=self.activeMask)
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            eraseMask = np.logical_not(self.brush_patch.to_mask(self.activeMask.shape))
            np.logical_and(self.activeMask, eraseMask, out=self.activeMask)
            if self.toolbox_window.get_erase_from_all_rois():
                np.logical_and(self.otherMask, eraseMask, out=self.otherMask)
        #self.do_reblit()

    # override from ImageShow
    def mouseMoveCB(self, event):
        self.fig.canvas.activateWindow()
        if (self.getState() == 'MUSCLE' and
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
        if self.getState() == 'MUSCLE':
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
            #print("Event outside")
            return

        if self.getState() != 'MUSCLE': return

        if self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK:
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
        if self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.saveSnapshot() # save state before modification
            if self.roiManager is not None:
                self.roiManager.set_mask(self.getCurrentROIName(), self.curImage, self.activeMask)
            if self.toolbox_window.get_erase_from_all_rois():
                for (key_tuple, mask) in self.roiManager.all_masks(image_number=self.curImage):
                    if key_tuple[0] == self.getCurrentROIName(): continue
                    self.roiManager.set_mask(key_tuple[0], key_tuple[1], np.logical_and(mask, self.otherMask))

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
            if pressed_key_without_modifiers == 'z':
                self.undo_signal.emit()
            elif pressed_key_without_modifiers == 'y':
                self.redo_signal.emit()
            return

        if event.key == 'n':
            self.propagate()
        elif event.key == 'b':
            self.propagateBack()
        elif event.key == '-' or event.key == 'y' or event.key == 'z':
            self.reduce_brush_size.emit()
        elif event.key == '+' or event.key == 'x':
            self.increase_brush_size.emit()
        elif event.key == 'r':
            self.roiRemoveOverlap()
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

        print("Saving ROIs", roiPickleName)
        if self.roiManager and not self.roiManager.is_empty():  # make sure ROIs are not empty
            dumpObj = {'classifications': self.classifications,
                       'roiManager': self.roiManager }
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
        print("Loading ROIs", roiPickleName)
        try:
            dumpObj = pickle.load(open(roiPickleName, 'rb'))
        except UnicodeDecodeError:
            print('Warning: Unicode decode error')
            dumpObj = pickle.load(open(roiPickleName, 'rb'), encoding='latin1')
        except:
            traceback.print_exc()
            self.alert("Unspecified error")
            return

        roiManager = None
        classifications = self.classifications

        if type(dumpObj) == ROIManager:
            roiManager = dumpObj
        elif type(dumpObj) == dict:
            try:
                classifications = dumpObj['classifications']
                roiManager = dumpObj['roiManager']
            except KeyError:
                self.alert("Unrecognized saved ROI type")
                return

        try:
            # print(self.allROIs)
            assert type(roiManager) == ROIManager
        except:
            self.alert("Unrecognized saved ROI type")
            return

        if roiManager.mask_size[0] != self.image.shape[0] or \
            roiManager.mask_size[1] != self.image.shape[1]:
            self.alert("ROI for wrong dataset")
            return

        print('Rois loaded')
        self.clearAllROIs()
        self.roiManager = roiManager
        available_classes = self.toolbox_window.get_available_classes()
        for i, classification in enumerate(classifications[:]):
            if classification not in available_classes:
                classifications[i] = 'None'

        self.classifications = classifications
        self.updateRoiList()
        self.updateMasksFromROIs()
        self.updateContourPainters()
        print("Contour updated")
        self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
        self.redraw()
        print("Redraw done")

    @pyqtSlot(str, str)
    @pyqtSlot(str)
    def loadDirectory(self, path, override_class=None):
        self.setSplash(True, 0, 1, "Loading dataset")

        def __reset_state():
            self.imList = []
            self.resetInternalState()
            self.override_class = override_class
            self.resolution_valid = False
            self.affine = None
            self.image = None
            self.resolution = [1, 1, 1]

        def __cleanup():
            __reset_state()
            self.setSplash(False)

        def __error(error = None):
            print(error, file=sys.stderr)
            self.alert("Error loading dataset. See the log for details")
            __cleanup()
            self.displayImage(None)
            self.redraw()

        __reset_state()
        _, ext = os.path.splitext(path)
        mask_dictionary = None
        if ext.lower() == '.npz':
            # data and mask bundle
            bundle = np.load(path, allow_pickle=False)
            if 'data' not in bundle:
                self.alert('No data in bundle!')
                self.setSplash(False, 1, 2, "")
                return
            if 'comment' in bundle:
                self.alert('Loading bundle with comment:\n' + str(bundle['comment']))

            self.basepath = os.path.dirname(path)
            try:
                self.loadNumpyArray(bundle['data'])
            except Exception as e:
                __error(e)
                return

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
                ImageShow.loadDirectory(self, path)
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
                self.axes.set_aspect(aspect=self.resolution[1]/self.resolution[0])

        # this is in case appendimage was never called
        if len(self.classifications) == 0:
            self.update_all_classifications()

        roi_bak_name = self.getRoiFileName() + '.' + datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            shutil.copyfile(self.getRoiFileName(), roi_bak_name)
        except:
            print("Warning: cannot copy roi file")

        self.roiManager = ROIManager(self.imList[0].shape)
        self.registrationManager = RegistrationManager(self.imList,
                                                       None,
                                                       os.getcwd(),
                                                       GlobalConfig['TEMP_DIR'])
        self.registrationManager.set_standard_transforms_name(self.basepath, self.basename)
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
    def saveResults(self, pathOut: str, outputType: str):
        # outputType is 'dicom', 'npy', 'npz', 'nifti', 'compact_dicom', 'compact_nifti'
        print("Saving results...")

        self.setSplash(True, 0, 4, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        self.setSplash(True, 1, 4, "Incremental learning...")

        # perform incremental learning
        if GlobalConfig['DO_INCREMENTAL_LEARNING']:
            self.incrementalLearn(dataForTraining, segForTraining, meanDiceScore, True)

        self.setSplash(True, 3, 4, "Saving file...")

        if outputType == 'dicom':
            save_dicom_masks(pathOut, allMasks, self.affine, self.dicomHeaderList)
        elif outputType == 'nifti':
            save_nifti_masks(pathOut, allMasks, self.affine)
        elif outputType == 'npy':
            save_npy_masks(pathOut, allMasks)
        elif outputType == 'compact_dicom':
            save_single_dicom_dataset(pathOut, allMasks, self.affine, self.dicomHeaderList)
        elif outputType == 'compact_nifti':
            save_single_nifti(pathOut, allMasks, self.affine)
        else: # assume the most generic outputType == 'npz':
            save_npz_masks(pathOut, allMasks)

        self.setSplash(False, 4, 4, "End")

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

    @pyqtSlot(str)
    def uploadData(self, comment = ''):
        print('Uploading data')
        dataset = self.getDatasetAsNumpy()
        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)
        resolution = np.array(self.resolution)
        out_data = {'data': dataset, 'resolution': resolution, 'comment': comment}
        for mask_name, mask in allMasks.items():
            out_data[f'mask_{mask_name}'] = mask
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
            self.alert(text)

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
            for index, name in enumerate(names):
                mask = np.zeros_like(accumulated_mask)
                mask[accumulated_mask == (index + 1)] = 1
                load_mask_validate(name, mask)

        def read_names_from_legend(legend_file):
            name_list = []
            with open(legend_file, newline='') as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)
                for row in reader:
                    name_list.append(row[1])
            return name_list


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
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False)
            name, _ = os.path.splitext(os.path.basename(path))

            mask = align_masks(mask_medical_volume)

            self.setSplash(True, 2, 3, "Importing masks")
            if mask.max() > 1: # dataset with multiple labels
                # try loading the legend
                legend_name = path + '.csv'
                try:
                    names = read_names_from_legend(legend_name)
                except FileNotFoundError:
                    fail('No legend file found')
                load_accumulated_mask(names, mask)
            else:
                load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return
        elif ext in dicom_ext:
            # load dicom masks
            path = os.path.dirname(path)
            mask_medical_volume, *_ = dosma_volume_from_path(path, reorient_data=False)
            name = os.path.basename(path)

            mask = align_masks(mask_medical_volume)
            self.setSplash(True, 2, 3, "Importing masks")
            if mask.max() > 1: # dataset with multiple labels
                # try loading the legend
                legend_name = os.path.join(path, 'legend.csv')
                try:
                    names = read_names_from_legend(legend_name)
                except FileNotFoundError:
                    fail('No legend file found')
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
                    mask_medical_volume, *_ = dosma_volume_from_path(data_path, reorient_data=False)
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



    ########################################################################################
    ###
    ### Deep learning functions
    ###
    ########################################################################################

    @pyqtSlot(str, str)
    def importModel(self, modelFile, modelName):
        if not isinstance(self.model_provider, LocalModelProvider):
            print('Trying to import a model in a non-local provider')
            return

        self.setSplash(True, 0, 1, 'Importing model...')

        try:
            self.model_provider.import_model(modelFile, modelName)
        except Exception as err:
            self.alert('Error while importing model. Probably invalid model')
            self.setSplash(False, 0, 1, 'Importing model...')
            traceback.print_exc()
            return
        self.setSplash(True, 1, 1, 'Importing model...')
        self.alert('Model imported successfully')
        self.setSplash(False, 1, 1, 'Importing model...')
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

    def setAvailableClasses(self, classList):
        try:
            classList.remove('Classifier')
        except ValueError: # Classifier doesn't exist. It doesn't matter
            pass

        new_class_list = []
        for c in classList:
            if self.model_provider is None:
                new_class_list.append(c)
            else:
                model_details = self.model_provider.model_details(c)
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

        for i, classification in enumerate(self.classifications[:]):
            if classification not in new_class_list:
                self.classifications[i] = 'None'
        self.toolbox_window.set_available_classes(new_class_list)
        try:
            self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
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

        for slice_number in range(min_slice, max_slice+1):
            self.displayImage(slice_number)
            self.doSegmentation()
            self.setSplash(True, 0, 3, "Loading model...")
            time.sleep(0.5)
        self.setSplash(False, 0, 3, "")

    def getSegmentedMasks(self, imIndex, setSplash=False, downloadModel=True):
        class_str = self.classifications[imIndex]
        if class_str == 'None':
            self.alert('Segmentation with "None" model is impossible!')
            return

        model_str = class_str.split(',')[0].strip()  # get the base model string in case of multiple variants.
        # variants are identified by "Model, Variant"

        if setSplash:
            self.setSplash(True, 0, 3, "Loading model...")

        try:
            segmenter = self.dl_segmenters[model_str]
        except KeyError:
            if downloadModel:
                if setSplash:
                    splashCallback = lambda cur_val, max_val: self.setSplash(True, cur_val, max_val,
                                                                                               'Downloading Model...')
                else:
                    splashCallback = None
                segmenter = self.model_provider.load_model(model_str, splashCallback,
                                                       force_download=GlobalConfig['FORCE_MODEL_DOWNLOAD'])
                if segmenter is None:
                    self.setSplash(False, 0, 3, "Loading model...")
                    self.alert(f"Error loading model {model_str}")
                    return None
                self.dl_segmenters[class_str] = segmenter
            else:
                return None

        if setSplash:
            self.setSplash(True, 1, 3, "Running segmentation...")
        inputData = {'image': self.imList[imIndex], 'resolution': self.resolution[0:2],
                     'split_laterality': GlobalConfig['SPLIT_LATERALITY'], 'classification': class_str}
        print("Segmenting image...")
        masks_out = segmenter(inputData)
        return masks_out

    @pyqtSlot()
    @snapshotSaver
    def doSegmentation(self):
        # run the segmentation
        imIndex = int(self.curImage)

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
        self.redraw()

    #@pyqtSlot()
    #@separate_thread_decorator # this crashes tensorflow!!
    @pyqtSlot()
    def incrementalLearnStandalone(self):
        self.setSplash(True, 0, 4, "Calculating maps...")
        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)
        self.setSplash(True, 1, 4, "Incremental learning...")
        # perform incremental learning
        self.incrementalLearn(dataForTraining, segForTraining, meanDiceScore, True)
        self.setSplash(False, 3, 4, "Saving file...")

    def incrementalLearn(self, dataForTraining, segForTraining, meanDiceScore, setSplash=False):
        performed = False
        for classification_name in dataForTraining:
            if classification_name == 'None': continue
            print(f'Performing incremental learning for {classification_name}')
            if len(dataForTraining[classification_name]) < GlobalConfig['IL_MIN_SLICES']:
                print(f"Not enough slices for {classification_name}")
                continue
            performed = True
            model_str = classification_name.split(',')[0].strip()  # get the base model string in case of multiple variants.
                                                        # variants are identified by "Model, Variant"
            try:
                model = self.dl_segmenters[model_str]
            except KeyError:
                model = self.model_provider.load_model(model_str, force_download=GlobalConfig['FORCE_MODEL_DOWNLOAD'])
                if model is None:
                    self.setSplash(False, 0, 3, "Loading model...")
                    self.alert(f"Error loading model {model_str}")
                    return
                self.dl_segmenters[model_str] = model
            training_data = []
            training_outputs = []
            for imageIndex in dataForTraining[classification_name]:
                training_data.append(dataForTraining[classification_name][imageIndex])
                training_outputs.append(segForTraining[classification_name][imageIndex])
                self.slicesUsedForTraining.add(imageIndex) # add the slice to the set of already used ones

            try:
                # todo: adapt bs and minTrainImages if needed
                model.incremental_learn({'image_list': training_data, 'resolution': self.resolution[0:2], 'classification': classification_name},
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