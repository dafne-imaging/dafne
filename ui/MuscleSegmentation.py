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

matplotlib.use("Qt5Agg")

import os, time, math

from config import GlobalConfig, load_config
load_config()

from .ToolboxWindow import ToolboxWindow
from .pyDicomView import ImageShow
from utils.mask_utils import calc_dice_score, save_npy_masks, save_npz_masks, save_dicom_masks, save_nifti_masks
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import shutil
from datetime import datetime
from utils.ROIManager import ROIManager

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import pickle
import os.path
from collections import deque
import functools
import csv
from utils.dicomUtils.dicom3D import load3dDicom
from utils.dicomUtils.alignDatasets import calcTransform, calcTransform2DStack

from utils.ThreadHelpers import separate_thread_decorator

from .BrushPatches import SquareBrush, PixelatedCircleBrush
from .ContourPainter import ContourPainter
import traceback

from dl.LocalModelProvider import LocalModelProvider
from dl.RemoteModelProvider import RemoteModelProvider

from utils.RegistrationManager import RegistrationManager

import requests

try:
    import SimpleITK as sitk # this requires simpleelastix! It is NOT available through PIP
except:
    sitk = None

try:
    import radiomics
except:
    radiomics = None

import re
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

    def __init__(self, *args, **kwargs):
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
            self.redraw()
        except:
            pass

    def resetInternalState(self):
        #load_config() # this was already loaded in dafne.py
        self.imList = []
        self.curImage = 0
        self.classifications = []
        self.originalSegmentationMasks = {}
        self.lastsave = datetime.now()

        self.roiChanged = {}
        self.history = deque(maxlen=GlobalConfig['HISTORY_LENGTH'])
        self.currentHistoryPoint = 0

        self.registrationManager = None

        self.resetModelProvider()
        self.resetInterface()

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

        self.toolbox_window.statistics_calc.connect(self.saveStats)
        self.toolbox_window.radiomics_calc.connect(self.saveRadiomics)

        self.toolbox_window.mask_import.connect(self.loadMask)

        self.splash_signal.connect(self.toolbox_window.set_splash)
        self.interface_disabled = False
        self.splash_signal.connect(self.disableInterface)

        self.toolbox_window.mask_grow.connect(self.maskGrow)
        self.toolbox_window.mask_shrink.connect(self.maskShrink)

        self.toolbox_window.config_changed.connect(self.configChanged)
        self.toolbox_window.data_upload.connect(self.uploadData)

        self.toolbox_window.model_import.connect(self.importModel)


    def setSplash(self, is_splash, current_value, maximum_value, text= ""):
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
    @separate_thread_decorator
    def changeEditMode(self, mode):
        print("Changing edit mode")
        self.setSplash(True, 0, 1)
        self.editMode = mode
        roi_name = self.getCurrentROIName()
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
                    self.addSubRoi(roiName, imageN)
                n_subrois = self.roiManager.get_roi_mask_pair(roiName, imageN).get_subroi_len()
            roiDict[roiName] = n_subrois  # dict: roiname -> n subrois per slice
        self.toolbox_window.set_rois_list(roiDict)
        self.updateContourPainters()
        self.updateMasksFromROIs()

    def alert(self, text):
        self.toolbox_window.alert(text)

    #############################################################################################
    ###
    ### History
    ###
    #############################################################################################

    def saveSnapshot(self):
        # clear history until the current point, so we can't redo anymore
        # print("Saving snapshot")
        while self.currentHistoryPoint > 0:
            self.history.popleft()
            self.currentHistoryPoint -= 1
        self.history.appendleft(pickle.dumps(self.roiManager))
        self.undo_possible.emit(self.canUndo())
        self.redo_possible.emit(self.canRedo())

    def canUndo(self):
        return self.currentHistoryPoint < len(self.history) - 1

    def canRedo(self):
        return self.currentHistoryPoint > 0

    def _changeHistory(self):
        #print(self.currentHistoryPoint, len(self.history))
        roiName = self.getCurrentROIName()
        subRoiNumber = self.getCurrentSubroiNumber()
        self.clearAllROIs()
        self.roiManager = pickle.loads(self.history[self.currentHistoryPoint])
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
            self.saveSnapshot()  # push current status into the history for redo
        self.currentHistoryPoint += 1
        self._changeHistory()

    @pyqtSlot()
    def redo(self):
        if not self.canRedo(): return
        self.currentHistoryPoint -= 1
        self._changeHistory()
        if self.currentHistoryPoint == 0:
            self.history.popleft()  # remove current status from the history

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
        self.redraw()

    def clearSubrois(self, name, sliceN):
        self.roiManager.clear(name, sliceN)
        self.updateRoiList()
        self.redraw()

    @pyqtSlot(str)
    @snapshotSaver
    def removeRoi(self, roi_name):
        self.roiManager.clear(roi_name)
        self.updateRoiList()
        self.redraw()

    @pyqtSlot(int)
    @snapshotSaver
    def removeSubRoi(self, subroi_number):
        current_name, _ = self.toolbox_window.get_current_roi_subroi()
        self.roiManager.clear_subroi(current_name, int(self.curImage), subroi_number)
        self.updateRoiList()
        self.redraw()

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
        self.redraw()

    @pyqtSlot()
    #@snapshotSaver this generates too many calls; anyway we want to add the subroi to the history
    # when something happens to it
    def addSubRoi(self, roi_name=None, imageN=None):
        if not roi_name:
            roi_name, _ = self.toolbox_window.get_current_roi_subroi()
        if imageN is None:
            imageN = int(self.curImage)
        self.roiManager.add_subroi(roi_name, imageN)
        self.updateRoiList()
        self.toolbox_window.set_current_roi(roi_name, self.roiManager.get_roi_mask_pair(roi_name,
                                                                                        imageN).get_subroi_len() - 1)
        self.redraw()

    @pyqtSlot(str, int)
    def changeRoi(self, roi_name, subroi_index):
        """ Change the active ROI """
        #print(roi_name, subroi_index)
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
        self.redraw()

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

        dataForTraining = {}
        segForTraining = {}

        roi_names = self.roiManager.get_roi_names()
        current_roi_index = 0

        for roiName in self.roiManager.get_roi_names():
            if setSplash:
                self.setSplash(True, current_roi_index, len(roi_names), "Calculating maps...")
                current_roi_index += 1
            masklist = []
            for imageIndex in range(len(self.imList)):
                roi = np.zeros(imSize)
                if self.roiManager.contains(roiName, imageIndex):
                    roi = self.roiManager.get_mask(roiName, imageIndex)
                masklist.append(roi)
                try:
                    originalSegmentation = self.originalSegmentationMasks[imageIndex][roiName]
                except:
                    originalSegmentation = None

                if originalSegmentation is not None:
                    diceScores.append(calc_dice_score(originalSegmentation, roi))
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

        diceScores = np.array(diceScores)
        print(diceScores)
        print("Average Dice score", diceScores.mean())
        return allMasks, dataForTraining, segForTraining, diceScores.mean()

    @pyqtSlot(str, str, bool)
    @snapshotSaver
    def copyRoi(self, originalName, newName, makeCopy=True):
        if makeCopy:
            self.roiManager.copy_roi(originalName, newName)
        else:
            self.roiManager.rename_roi(originalName, newName)
        self.updateRoiList()

    @pyqtSlot(str, str, str, str)
    @snapshotSaver
    def combineRoi(self, roi1, roi2, operator, dest_roi):
        if operator == 'Union':
            combine_fn = np.logical_or
        elif operator == 'Subtraction':
            combine_fn = lambda x,y: np.logical_and(x, np.logical_not(y))
        elif operator == 'Intersection':
            combine_fn = np.logical_and
        elif operator == 'Exclusion':
            combine_fn = np.logical_xor
        self.roiManager.generic_roi_combine(roi1, roi2, combine_fn, dest_roi)
        self.updateMasksFromROIs()
        self.updateContourPainters()
        self.updateRoiList()

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
        # self.redraw()
        self.redraw()

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
        # self.redraw()
        self.redraw()

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
        # self.redraw()
        self.redraw()

    # No @snapshotSaver: snapshot is saved in the calling function
    def movePoint(self, spline, event):
        if self.currentPoint is None:
            return
        spline.replaceKnot(self.currentPoint, (event.xdata, event.ydata))
        # self.redraw()
        self.redraw()

    @pyqtSlot()
    @snapshotSaver
    def clearCurrentROI(self):
        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            roi = self.getCurrentROI()
            roi.removeAllKnots()
        elif self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.roiManager.clear_mask(self.getCurrentROIName(), self.curImage)
            self.activeMask = None
        self.redraw()

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
        self.redraw()

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
    @separate_thread_decorator
    def propagate(self):
        if self.curImage >= len(self.imList) - 1: return
        if not self.registrationManager: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]

        self.setSplash(True, 0, 3)


        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            curROI = self.getCurrentROI()
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
    @separate_thread_decorator
    def propagateBack(self):
        if self.curImage < 1: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]

        self.setSplash(True, 0, 3)

        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            curROI = self.getCurrentROI()
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
        if self.activeMask is None or self.otherMask is None:
            self.updateMasksFromROIs()

        if not self.hideRois:  # if we hide the ROIs, clear all the masks
            active_mask = self.activeMask
            other_mask = self.otherMask
        else:
            active_mask = np.zeros_like(self.activeMask)
            other_mask = np.zeros_like(self.otherMask)

        if self.maskImPlot is None:
            self.maskImPlot = self.axes.imshow(active_mask, cmap=self.mask_layer_colormap, alpha=GlobalConfig['MASK_LAYER_ALPHA'], vmin=0, vmax=1, zorder=100)

        self.maskImPlot.set_data(active_mask.astype(np.uint8))

        if self.maskOtherImPlot is None:
            relativeAlphaROI = GlobalConfig['ROI_OTHER_COLOR'][3] / GlobalConfig['ROI_COLOR'][3]
            self.maskOtherImPlot = self.axes.imshow(other_mask, cmap=self.mask_layer_other_colormap, alpha=relativeAlphaROI*GlobalConfig['MASK_LAYER_ALPHA'], vmin=0, vmax=1, zorder=101)

        self.maskOtherImPlot.set_data(other_mask.astype(np.uint8))

    def updateContourPainters(self):
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

    def displayImage(self, im, cmap=None):
        self.removeMasks()
        self.removeContours()
        ImageShow.displayImage(self, im, cmap)
        self.updateRoiList()  # set the appropriate (sub)roi list for the current image
        self.activeMask = None
        self.otherMask = None
        self.updateContourPainters()
        self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo

    ##############################################################################################################
    ###
    ### UI Callbacks
    ###
    ##############################################################################################################

    @pyqtSlot()
    def refreshCB(self):
        # check if ROIs should be autosaved
        now = datetime.now()
        if (now - self.lastsave).total_seconds() > GlobalConfig['AUTOSAVE_INTERVAL']:
            self.lastsave = now
            self.saveROIPickle()

        if not self.app:
            app = QApplication.instance()

        if self.wacom:
            app.setOverrideCursor(Qt.BlankCursor)
        else:
            app.setOverrideCursor(Qt.ArrowCursor)

        #print("Refresh")
        #print(self.editMode)

        if self.roiManager:
            if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
                self.drawContours()
            elif self.editMode == ToolboxWindow.EDITMODE_MASK:
                self.drawMasks()

        #print("Redrawing")
        #print(self.axes.get_children())
        #plt.draw() - already in redraw

    def closeCB(self, event):
        self.toolbox_window.close()
        if not self.basepath: return
        if self.registrationManager:
            self.registrationManager.pickle_transforms()
        self.saveROIPickle()

    def moveBrushPatch(self, event):
        """
            moves the brush. Returns True if the brush was moved to a new position
        """
        brush_type, brush_size = self.toolbox_window.get_brush()
        mouseX = event.xdata
        mouseY = event.ydata
        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
            brush_color = GlobalConfig['BRUSH_PAINT_COLOR']
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            brush_color = GlobalConfig['BRUSH_ERASE_COLOR']
        else:
            brush_color = None
        if mouseX is None or mouseY is None or brush_color is None:
            try:
                self.brush_patch.remove()
                self.fig.canvas.draw()
            except:
                pass
            self.brush_patch = None
            return False

        try:
            oldX = self.moveBrushPatch_oldX  # static variables
            oldY = self.moveBrushPatch_oldY
        except:
            oldX = -1
            oldY = -1

        mouseX = np.round(mouseX)
        mouseY = np.round(mouseY)

        if oldX == mouseX and oldY == mouseY:
            return False

        self.moveBrushPatch_oldX = mouseX
        self.moveBrushPatch_oldY = mouseY

        if brush_type == ToolboxWindow.BRUSH_SQUARE:
            xy = (math.floor(mouseX - brush_size / 2) + 0.5, math.floor(mouseY - brush_size / 2) + 0.5)
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
            center = (math.floor(mouseX), math.floor(mouseY))
            if type(self.brush_patch) != PixelatedCircleBrush:
                try:
                    self.brush_patch.remove()
                except:
                    pass
                self.brush_patch = PixelatedCircleBrush(center, brush_size, color=brush_color)
                self.axes.add_patch(self.brush_patch)

            self.brush_patch.set_center(center)
            self.brush_patch.set_radius(brush_size)

        self.brush_patch.set_color(brush_color)
        self.fig.canvas.draw()
        return True

    def modifyMaskFromBrush(self, saveSnapshot=False):
        if not self.brush_patch: return
        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
            if saveSnapshot: self.saveSnapshot()
            np.logical_or(self.activeMask, self.brush_patch.to_mask(self.activeMask.shape), out=self.activeMask)
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            if saveSnapshot: self.saveSnapshot()
            eraseMask = np.logical_not(self.brush_patch.to_mask(self.activeMask.shape))
            np.logical_and(self.activeMask, eraseMask, out=self.activeMask)
            if self.toolbox_window.get_erase_from_all_rois():
                np.logical_and(self.otherMask, eraseMask, out=self.otherMask)
        self.redraw()

    # override from ImageShow
    def mouseMoveCB(self, event):
        if (self.getState() == 'MUSCLE' and
                self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK and
                self.isCursorNormal() and
                event.button != 2 and
                event.button != 3):
            moved_to_new_point = self.moveBrushPatch(event)
            if event.button == 1: # because we are overriding MoveCB, we won't call leftPressCB
                if moved_to_new_point:
                    self.modifyMaskFromBrush()
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
                self.redraw()
            elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.ROTATE_STATE:
                if self.rotationDelta is None: return
                newAngle = roi.getOrientation( (event.xdata, event.ydata), center = self.rotationDelta[0])
                roi.reorientByAngle(newAngle - self.rotationDelta[1])
                self.redraw()

    def leftPressCB(self, event):
        if not self.imPlot.contains(event):
            print("Event outside")
            return

        if self.getState() != 'MUSCLE': return

        if self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK:
            self.modifyMaskFromBrush(saveSnapshot=True)
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
                    # self.redraw()
                    self.redraw()
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

    def keyPressCB(self, event):
        # print(event.key)
        if 'shift' in event.key:
            self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.ADD_STATE)
        elif 'control' in event.key or 'cmd' in event.key or 'super' in event.key or 'ctrl' in event.key:
            self.toolbox_window.set_temp_edit_button_state(ToolboxWindow.REMOVE_STATE)
        if event.key == 'n':
            self.propagate()
        elif event.key == 'b':
            self.propagateBack()
        else:
            ImageShow.keyPressCB(self, event)

    def keyReleaseCB(self, event):
        if 'shift' in event.key or 'control' in event.key or 'cmd' in event.key or 'super' in event.key or 'ctrl' in event.key:
            self.toolbox_window.restore_edit_button_state()

        # plt.show()

    ################################################################################################################
    ###
    ### I/O
    ###
    ################################################################################################################

    def getDatasetAsNumpy(self):
        return np.transpose(np.stack(self.imList), [1,2,0])

    @pyqtSlot(str)
    def saveROIPickle(self, roiPickleName=None):
        showWarning = True
        if not roiPickleName:
            roiPickleName = self.getRoiFileName()
            showWarning = False # don't show a empty roi warning if autosaving
        print("Saving ROIs", roiPickleName)
        if self.roiManager and not self.roiManager.is_empty():  # make sure ROIs are not empty
            dumpObj = {'classifications': self.classifications,
                       'roiManager': self.roiManager }
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
        self.toolbox_window.set_class(self.classifications[int(self.curImage)])  # update the classification combo
        self.redraw()

    @pyqtSlot(str, str)
    @pyqtSlot(str)
    def loadDirectory(self, path, override_class=None):
        self.imList = []
        self.resetInternalState()
        self.override_class = override_class
        ImageShow.loadDirectory(self, path)
        roi_bak_name = self.getRoiFileName() + '.' + datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            shutil.copyfile(self.getRoiFileName(), roi_bak_name)
        except:
            print("Warning: cannot copy roi file")

        self.roiManager = ROIManager(self.imList[0].shape)
        self.registrationManager = RegistrationManager(self.imList,
                                                       os.path.join(self.basepath, 'transforms.p'),
                                                       os.getcwd(),
                                                       GlobalConfig['TEMP_DIR'])
        #self.loadROIPickle()
        self.redraw()
        self.toolbox_window.general_enable(True)
        self.toolbox_window.set_exports_enabled(numpy= True,
                                                dicom= (self.dicomHeaderList is not None),
                                                nifti= (self.affine is not None)
                                                )

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
        # outputType is 'dicom', 'npy', 'npz', 'nifti'
        print("Saving results...")

        self.setSplash(True, 0, 4, "Calculating maps...")

        allMasks, dataForTraining, segForTraining, meanDiceScore = self.calcOutputData(setSplash=True)

        self.setSplash(True, 1, 4, "Incremental learning...")

        # perform incremental learning
        if GlobalConfig['DO_INCREMENTAL_LEARNING']:
            for classification_name in dataForTraining:
                if classification_name == 'None': continue
                print(f'Performing incremental learning for {classification_name}')
                try:
                    model = self.dl_segmenters[classification_name]
                except KeyError:
                    model = self.model_provider.load_model(classification_name)
                    self.dl_segmenters[classification_name] = model
                training_data = []
                training_outputs = []
                for imageIndex in dataForTraining[classification_name]:
                    training_data.append(dataForTraining[classification_name][imageIndex])
                    training_outputs.append(segForTraining[classification_name][imageIndex])

                try:
                    #todo: adapt bs and minTrainImages if needed
                    model.incremental_learn({'image_list': training_data, 'resolution': self.resolution[0:2]},
                                                training_outputs, bs=5, minTrainImages=5)
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
                self.setSplash(True, 2, 4, "Sending the improved model to server...")

                st = time.time()
                if meanDiceScore is None:
                    meanDiceScore = -1.0
                self.model_provider.upload_model(classification_name, model, meanDiceScore)
                print(f"took {time.time() - st:.2f}s")

        self.setSplash(True, 3, 4, "Saving file...")

        if outputType == 'dicom':
            save_dicom_masks(pathOut, allMasks, self.dicomHeaderList)
        elif outputType == 'nifti':
            save_nifti_masks(pathOut, allMasks, self.affine, self.transpose)
        elif outputType == 'npy':
            save_npy_masks(pathOut, allMasks)
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
            csvRow = {}
            csvRow['roi_name'] = roi_name
            mask = roi_mask > 0
            masked = np.ma.array(dataset, mask=np.logical_not(roi_mask))
            csvRow['voxels'] = mask.sum()
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
            firstDicom = None
            for element in os.listdir(path):
                if element.startswith('.'): continue
                new_path = os.path.join(path, element)
                if os.path.isdir(new_path):
                    containsDirs = True
                else: # check if the folder contains dicoms
                    _, ext2 = os.path.splitext(new_path)
                    if ext2.lower() in dicom_ext:
                        containsDicom = True
                        if firstDicom is None:
                            firstDicom = new_path

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
            if mask.shape[0] != self.image.shape[0] or mask.shape[1] != self.image.shape[1]:
                fail("Mask size mismatch")
                return
            if mask.ndim > 2:
                is3D = True
                if mask.shape[2] != len(self.imList):
                    fail("Mask size mismatch")
                    return
            mask = mask > 0
            self.masksToRois({name: mask}, int(self.curImage)) # this is OK for 2D and 3D

        def is_local_stack_2d():
            try:
                # in a 3D dataset, the following is not defined
                spacing = self.dicomHeaderList[0].SpacingBetweenSlices
                print("Aligning 2D Stack")
                return True
            except:
                print("Aligning 3D Stack")
                return False

        def align_dicom(dataset, dicomInfo):
            # check if 1) we have dicom headers to align the dataset and 2) the datasets are not already aligned
            if self.dicomHeaderList or not \
                    (all(self.dicomHeaderList[0].ImageOrientationPatient == dicomInfo[0].ImageOrientationPatient) \
                     and all(self.dicomHeaderList[0].ImagePositionPatient == dicomInfo[0].ImagePositionPatient)):
                # align datasets
                # find out if the loaded dataset is 3D or 2D stack
                self.setSplash(True, 1, 3, "Performing alignment")
                if is_local_stack_2d():
                    transform = calcTransform2DStack(None, self.dicomHeaderList, None, dicomInfo)
                else:
                    transform = calcTransform(None, self.dicomHeaderList, None, dicomInfo, False)
                mask = transform(dataset, maskMode=True)
            else:
                # we cannot align the datasets
                mask = dataset.squeeze()
            return mask

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
            fail("Nii masks not supported for loading")
        elif ext in dicom_ext:
            # load dicom masks
            path = os.path.dirname(path)
            dataset, dicomInfo = load3dDicom(path)
            name = os.path.basename(path)

            mask = align_dicom(dataset, dicomInfo)

            self.setSplash(True, 2, 3, "Importing masks")
            load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return
        elif ext == 'multidicom':
            # load multiple dicom masks and align them at the same time
            accumulated_mask = None
            current_mask_number = 1
            dicom_info_ok = None
            names = []
            for subdir in sorted(os.listdir(path)):
                if subdir.startswith('.'): continue
                subdir_path = os.path.join(path, subdir)
                if not os.path.isdir(subdir_path): continue
                dataset, dicomInfo = load3dDicom(subdir_path) # load the next dataset
                if dataset is None: continue
                dataset[dataset > 0] = 1
                dataset[dataset < 1] = 0
                name = os.path.basename(subdir_path)
                if accumulated_mask is None:
                    accumulated_mask = np.copy(dataset.astype(np.uint8))
                else:
                    try:
                        accumulated_mask += dataset.astype(np.uint8)*current_mask_number
                    except:
                        print('Incompatible mask')
                        continue
                names.append(name)
                #print("Mask number", current_mask_number, "accumulated max", accumulated_mask.max())
                current_mask_number += 1
                dicom_info_ok = dicomInfo
            if len(names) == 0:
                self.alert('No available mask found!')
                return

            aligned_masks = align_dicom(accumulated_mask, dicom_info_ok)

            self.setSplash(True, 2, 3, "Importing masks")
            for index, name in enumerate(names):
                mask = np.zeros_like(aligned_masks)
                mask[aligned_masks == (index+1)] = 1
                load_mask_validate(name, mask)
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
                self.dl_classifier = modelProvider.load_model('Classifier')
            except:
                self.dl_classifier = None
        else:
            self.dl_classifier = None

    def setAvailableClasses(self, classList):
        try:
            classList.remove('Classifier')
        except ValueError: # Classifier doesn't exist. It doesn't matter
            pass
        for i, classification in enumerate(self.classifications[:]):
            if classification not in classList:
                self.classifications[i] = 'None'
        self.toolbox_window.set_available_classes(classList)
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
            time.sleep(0.5)

    @pyqtSlot()
    @snapshotSaver
    def doSegmentation(self):
        # run the segmentation
        imIndex = int(self.curImage)
        class_str = self.classifications[imIndex]
        if class_str == 'None':
            self.alert('Segmentation with "None" model is impossible!')
            return
        self.setSplash(True, 0, 3, "Loading model...")
        try:
            segmenter = self.dl_segmenters[class_str]
        except KeyError:
            segmenter = self.model_provider.load_model(class_str, lambda cur_val,max_val: self.setSplash(True, cur_val, max_val, 'Downloading Model...'))
            if segmenter is None:
                self.setSplash(False, 0, 3, "Loading model...")
                self.alert(f"Error loading model {class_str}")
                return
            self.dl_segmenters[class_str] = segmenter

        self.setSplash(True, 1, 3, "Running segmentation...")
        t = time.time()
        inputData = {'image': self.imList[imIndex], 'resolution': self.resolution[0:2], 'split_laterality': GlobalConfig['SPLIT_LATERALITY']}
        print("Segmenting image...")
        masks_out = segmenter(inputData)
        self.originalSegmentationMasks[imIndex] = masks_out # save original segmentation for statistics
        self.setSplash(True, 2, 3, "Converting masks...")
        print("Done")
        self.masksToRois(masks_out, imIndex)
        self.activeMask = None
        self.otherMask = None
        print("Segmentation/import time:", time.time() - t)
        self.setSplash(False, 3, 3)
        self.redraw()
