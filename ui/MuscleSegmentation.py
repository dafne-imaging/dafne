#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:14:33 2020

@author: francesco

"""
import matplotlib

matplotlib.use("Qt5Agg")

import sys, os, time, math

# print(sys.path)
#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(os.path.normpath(SCRIPT_DIR))

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

try:
    import SimpleITK as sitk # this requires simpleelastix! It is NOT available through PIP
except:
    pass

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

DO_INCREMENTAL_LEARNING = True

ROI_CIRCLE_SIZE = 2
SIMPLIFIED_ROI_POINTS = 20
SIMPLIFIED_ROI_SPACING = 15
ROI_COLOR_ORIG = (1, 0, 0, 0.5)  # red with 0.5 opacity
ROI_SAME_COLOR_ORIG = (1, 1, 0, 0.5)  # yellow with 0.5 opacity
ROI_OTHER_COLOR_ORIG = (0, 0, 1, 0.4)

ROI_COLOR_WACOM = (1, 0, 0, 1)  # red with 1 opacity
ROI_SAME_COLOR_WACOM = (1, 1, 0, 1)  # yellow with 0.5 opacity
ROI_OTHER_COLOR_WACOM = (0, 0, 1, 0.8)

ROI_COLOR = ROI_COLOR_ORIG
ROI_SAME_COLOR = ROI_SAME_COLOR_ORIG
ROI_OTHER_COLOR = ROI_OTHER_COLOR_ORIG

BRUSH_PAINT_COLOR = (1, 0, 0, 0.6)
BRUSH_ERASE_COLOR = (0, 0, 1, 0.6)


ROI_FILENAME = 'rois.p'
AUTOSAVE_INTERVAL = 30

HIDE_ROIS_RIGHTCLICK = True

COLORS = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', 'indigo', 'white', 'grey']

HISTORY_LENGTH = 20

MASK_LAYER_COLORMAP = matplotlib.colors.ListedColormap(np.array([
    [0,0,0,0],
    [*ROI_COLOR[:3],1]
]))

MASK_LAYER_OTHER_COLORMAP = matplotlib.colors.ListedColormap(np.array([
    [0,0,0,0],
    [*ROI_OTHER_COLOR[:3],1]
]))


MASK_LAYER_ALPHA = 0.4

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

        self.wacom = False
        self.roiColor = ROI_COLOR
        self.roiOther = ROI_OTHER_COLOR
        self.roiSame = ROI_SAME_COLOR

        self.saveDicom = False

        self.model_provider = None
        self.dl_classifier = None
        self.dl_segmenters = {}


        # self.fig.canvas.setCursor(Qt.BlankCursor)
        self.app = None

        # self.setCmap('viridis')
        self.extraOutputParams = []
        self.transformsChanged = False

        self.hideRois = False
        self.editMode = ToolboxWindow.EDITMODE_MASK
        self.resetInternalState()

    def resetInternalState(self):
        self.imList = []
        self.curImage = 0
        self.classifications = []
        self.originalSegmentationMasks = {}
        self.lastsave = datetime.now()

        self.roiChanged = {}
        self.history = deque(maxlen=HISTORY_LENGTH)
        self.currentHistoryPoint = 0
        self.transforms = {}
        self.invtransforms = {}

        try:
            self.brush_patch.remove()
        except:
            pass

        try:
            self.removeMasks()
        except:
            pass
        self.brush_patch = None
        self.maskImPlot = None
        self.maskOtherImPlot = None
        self.activeMask = None
        self.otherMask = None

        try:
            self.removeContours()
        except:
            pass
        self.activeRoiPainter = ContourPainter(self.roiColor, ROI_CIRCLE_SIZE)
        self.sameRoiPainter = ContourPainter(self.roiSame, 0.1)
        self.otherRoiPainter = ContourPainter(self.roiOther, 0.1)

        self.roiManager = None


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


        self.toolbox_window = ToolboxWindow(activate_registration=showRegistrationGui)
        self.toolbox_window.show()

        self.toolbox_window.editmode_changed.connect(self.changeEditMode)

        self.toolbox_window.roi_added.connect(self.addRoi)
        self.toolbox_window.subroi_added.connect(self.addSubRoi)

        self.toolbox_window.roi_deleted.connect(self.removeRoi)
        self.toolbox_window.subroi_deleted.connect(self.removeSubRoi)

        self.toolbox_window.roi_changed.connect(self.changeRoi)

        self.toolbox_window.roi_clear.connect(self.clearCurrentROI)

        self.toolbox_window.do_autosegment.connect(self.doSegmentation)

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

        self.toolbox_window.statistics_calc.connect(self.saveStats)

        self.toolbox_window.mask_import.connect(self.loadMask)

        self.splash_signal.connect(self.toolbox_window.set_splash)
        self.splash_signal.connect(self.disableInterface)


    def setSplash(self, is_splash, current_value, maximum_value, text= ""):
        self.splash_signal.emit(is_splash, current_value, maximum_value, text)

    #dis/enable interface callbacks
    @pyqtSlot(bool, int, int, str)
    def disableInterface(self, disable, unused1, unused2, txt):
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

    def alert(self, text):
        self.toolbox_window.alert(text)

    #############################################################################################
    ###
    ### History
    ###
    #############################################################################################

    def saveSnapshot(self):
        # clear history until the current point, so we can't redo anymore
        print("Saving snapshot")
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
            roi_fname = self.basename + '.' + ROI_FILENAME
        else:
            roi_fname = ROI_FILENAME
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
                    print(diceScores)

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

    #########################################################################################
    ###
    ### ROI modifications
    ###
    #########################################################################################

    @snapshotSaver
    def simplify(self):
        r = self.getCurrentROI()
        # self.setCurrentROI(r.getSimplifiedSpline(SIMPLIFIED_ROI_POINTS))
        # self.setCurrentROI(r.getSimplifiedSpline(spacing=SIMPLIFIED_ROI_SPACING))
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
        print(minDeriv)
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


    #####################################################################################################
    ###
    ### Elastix
    ###
    #####################################################################################################

    def getInverseTransform(self, imIndex):
        try:
            return self.invtransforms[imIndex]
        except KeyError:
            self.calcInverseTransform(imIndex)
            return self.invtransforms[imIndex]

    def getTransform(self, imIndex):
        try:
            return self.transforms[imIndex]
        except KeyError:
            self.calcTransform(imIndex)
            return self.transforms[imIndex]

    def calcTransform(self, imIndex):
        if imIndex >= len(self.imList) - 1: return
        fixedImage = self.imList[imIndex]
        movingImage = self.imList[imIndex + 1]
        self.transforms[imIndex] = self.runElastix(fixedImage, movingImage)
        self.transformsChanged = True

    def calcInverseTransform(self, imIndex):
        if imIndex < 1: return
        fixedImage = self.imList[imIndex]
        movingImage = self.imList[imIndex - 1]
        self.invtransforms[imIndex] = self.runElastix(fixedImage, movingImage)
        self.transformsChanged = True

    def runElastix(self, fixedImage, movingImage):
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetLogToFile(False)

        elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixedImage))
        elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(movingImage))
        print("Registering...")

        elastixImageFilter.Execute()
        print("Done")
        pMap = elastixImageFilter.GetTransformParameterMap()
        self.cleanElastixFiles()
        return pMap

    def calcTransforms(self):
        qbar = QProgressBar()
        qbar.setRange(0, len(self.imList) - 1)
        qbar.setWindowTitle(QString("Registering images"))
        qbar.setWindowModality(Qt.ApplicationModal)
        qbar.move(800, 500)
        qbar.show()

        for imIndex in range(len(self.imList)):
            qbar.setValue(imIndex)
            plt.pause(.000001)
            print("Calculating image:", imIndex)
            # the transform was already calculated
            if imIndex not in self.transforms:
                self.calcTransform(imIndex)
            if imIndex not in self.invtransforms:
                self.calcInverseTransform(imIndex)

        qbar.close()
        print("Saving transforms")
        self.pickleTransforms()

    def propagateAll(self):
        while self.curImage < len(self.imList) - 1:
            self.propagate()
            plt.pause(.000001)

    def propagateBackAll(self):
        while self.curImage > 0:
            self.propagateBack()
            plt.pause(.000001)

    def cleanElastixFiles(self):
        files_to_delete = ['TransformixPoints.txt',
                           'outputpoints.txt',
                           'TransformParameters.0.txt',
                           'TransformParameters.1.txt',
                           'TransformParameters.2.txt']

        for file in files_to_delete:
            try:
                os.remove(file)
            except:
                pass


    def runTransformixMask(self, mask, transform):
        transformixImageFilter = sitk.TransformixImageFilter()

        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetLogToFile(False)

        for t in transform:
            t['ResampleInterpolator'] = ["FinalNearestNeighborInterpolator"]

        transformixImageFilter.SetTransformParameterMap(transform)

        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(mask))
        transformixImageFilter.Execute()

        mask_out = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        self.cleanElastixFiles()

        return mask_out.astype(np.uint8)

    def runTransformixKnots(self, knots, transform):
        transformixImageFilter = sitk.TransformixImageFilter()

        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetLogToFile(False)

        transformixImageFilter.SetTransformParameterMap(transform)

        # create Transformix point file
        with open("TransformixPoints.txt", "w") as f:
            f.write("point\n")
            f.write("%d\n" % (len(knots)))
            for k in knots:
                # f.write("%.3f %.3f\n" % (k[0], k[1]))
                f.write("%.3f %.3f\n" % (k[0], k[1]))

        transformixImageFilter.SetFixedPointSetFileName("TransformixPoints.txt")
        transformixImageFilter.SetOutputDirectory(".")
        transformixImageFilter.Execute()

        outputCoordRE = re.compile("OutputPoint\s*=\s*\[\s*([\d.]+)\s+([\d.]+)\s*\]")

        knotsOut = []

        with open("outputpoints.txt", "r") as f:
            for line in f:
                m = outputCoordRE.search(line)
                knot = (float(m.group(1)), float(m.group(2)))
                knotsOut.append(knot)

        self.cleanElastixFiles()

        return knotsOut

    @snapshotSaver
    @separate_thread_decorator
    def propagate(self):
        if self.curImage >= len(self.imList) - 1: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]

        self.setSplash(True, 0, 3)


        if self.editMode == ToolboxWindow.EDITMODE_CONTOUR:
            curROI = self.getCurrentROI()
            nextROI = self.getCurrentROI(+1)
            knotsOut = self.runTransformixKnots(curROI.knots, self.getTransform(int(self.curImage)))

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
            mask_out = self.runTransformixMask(mask_in, self.getInverseTransform(int(self.curImage+1)))
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
            knotsOut = self.runTransformixKnots(curROI.knots, self.getInverseTransform(int(self.curImage)))

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
            mask_out = self.runTransformixMask(mask_in, self.getTransform(int(self.curImage-1)))
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
        print('Removing masks')
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
            self.maskImPlot = self.axes.imshow(active_mask, cmap=MASK_LAYER_COLORMAP, alpha=MASK_LAYER_ALPHA, vmin=0, vmax=1, zorder=100)

        self.maskImPlot.set_data(active_mask)

        if self.maskOtherImPlot is None:
            self.maskOtherImPlot = self.axes.imshow(other_mask, cmap=MASK_LAYER_OTHER_COLORMAP, alpha=MASK_LAYER_ALPHA, vmin=0, vmax=1, zorder=101)

        self.maskOtherImPlot.set_data(other_mask)

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
        try:
            self.maskImPlot.remove()
        except:
            pass
        try:
            self.maskOtherImPlot.remove()
        except:
            pass
        self.maskImPlot = None
        self.maskOtherImPlot = None
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
        if (now - self.lastsave).total_seconds() > AUTOSAVE_INTERVAL:
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
        if not self.basepath: return
        self.toolbox_window.close()
        if self.transformsChanged: self.pickleTransforms()
        self.saveROIPickle()

    def moveBrushPatch(self, event):
        """
            moves the brush. Returns True if the brush was moved to a new position
        """
        brush_type, brush_size = self.toolbox_window.get_brush()
        mouseX = event.xdata
        mouseY = event.ydata
        if self.toolbox_window.get_edit_button_state() == ToolboxWindow.ADD_STATE:
            brush_color = BRUSH_PAINT_COLOR
        elif self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
            brush_color = BRUSH_ERASE_COLOR
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
            center = (math.floor(mouseX) + 0.5, math.floor(mouseY) + 0.5)
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
            np.logical_and(self.activeMask, np.logical_not(self.brush_patch.to_mask(self.activeMask.shape)),
                           out=self.activeMask)
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

    def leftPressCB(self, event):
        if not self.imPlot.contains(event):
            print("Event outside")
            return

        if self.getState() != 'MUSCLE': return

        if self.toolbox_window.get_edit_mode() == ToolboxWindow.EDITMODE_MASK:
            self.modifyMaskFromBrush(saveSnapshot=True)
        else:
            roi = self.getCurrentROI()
            knotIndex, knot = roi.findKnotEvent(event)
            if self.toolbox_window.get_edit_button_state() == ToolboxWindow.REMOVE_STATE:
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
        if self.editMode == ToolboxWindow.EDITMODE_MASK:
            self.roiManager.set_mask(self.getCurrentROIName(), self.curImage, self.activeMask)

    def rightPressCB(self, event):
        self.hideRois = HIDE_ROIS_RIGHTCLICK
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

    @pyqtSlot(str)
    def saveROIPickle(self, roiPickleName=None):
        if not roiPickleName:
            roiPickleName = self.getRoiFileName()
        print("Saving ROIs", roiPickleName)
        if self.roiManager and not self.roiManager.is_empty():  # make sure ROIs are not empty
            pickle.dump(self.roiManager, open(roiPickleName, 'wb'))

    @pyqtSlot(str)
    def loadROIPickle(self, roiPickleName=None):
        if not roiPickleName:
            roiPickleName = self.getRoiFileName()
        print("Loading ROIs", roiPickleName)
        try:
            roiManager = pickle.load(open(roiPickleName, 'rb'))
        except UnicodeDecodeError:
            print('Warning: Unicode decode error')
            roiManager = pickle.load(open(roiPickleName, 'rb'), encoding='latin1')
        except:
            self.alert("Unspecified error")
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
        self.updateRoiList()

    @pyqtSlot(str)
    def loadDirectory(self, path):
        self.imList = []
        self.resetInternalState()
        ImageShow.loadDirectory(self, path)
        roi_bak_name = self.getRoiFileName() + '.' + datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            shutil.copyfile(self.getRoiFileName(), roi_bak_name)
        except:
            print("Warning: cannot copy roi file")

        self.roiManager = ROIManager(self.imList[0].shape)
        self.unPickleTransforms()
        #self.loadROIPickle()
        self.redraw()
        self.toolbox_window.set_exports_enabled(numpy= True,
                                                dicom= (self.dicomHeaderList is not None),
                                                nifti= (self.affine is not None)
                                                )

    def appendImage(self, im):
        ImageShow.appendImage(self, im)
        print("new Append Image")
        if not self.dl_classifier: return
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
        if DO_INCREMENTAL_LEARNING:
            for classification_name in dataForTraining:
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
                model.incremental_learn({'image_list': training_data, 'resolution': self.resolution[0:2]}, training_outputs)
                print('Done')

            self.setSplash(True, 2, 4, "Sending the improved model...")
            #TODO: send the model back to the server

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

        dataset = np.transpose(np.stack(self.imList), [1,2,0])

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



    def pickleTransforms(self):
        if not self.basepath: return
        pickleObj = {}
        transformDict = {}
        for k, transformList in self.transforms.items():
            curTransformList = []
            for transform in transformList:
                curTransformList.append(transform.asdict())
            transformDict[k] = curTransformList
        invTransformDict = {}
        for k, transformList in self.invtransforms.items():
            curTransformList = []
            for transform in transformList:
                curTransformList.append(transform.asdict())
            invTransformDict[k] = curTransformList
        pickleObj['direct'] = transformDict
        pickleObj['inverse'] = invTransformDict
        outFile = os.path.join(self.basepath, 'transforms.p')
        pickle.dump(pickleObj, open(outFile, 'wb'))

    def unPickleTransforms(self):
        if not self.basepath: return
        pickleFile = os.path.join(self.basepath, 'transforms.p')
        try:
            pickleObj = pickle.load(open(pickleFile, 'rb'))
        except:
            print("Error trying to load transforms")
            return

        transformDict = pickleObj['direct']
        self.transforms = {}
        for k, transformList in transformDict.items():
            curTransformList = []
            for transform in transformList:
                curTransformList.append(sitk.ParameterMap(transform))
            self.transforms[k] = tuple(curTransformList)
        invTransformDict = pickleObj['inverse']
        self.invtransforms = {}
        for k, transformList in invTransformDict.items():
            curTransformList = []
            for transform in transformList:
                curTransformList.append(sitk.ParameterMap(transform))
            self.invtransforms[k] = tuple(curTransformList)

    @pyqtSlot(str)
    @separate_thread_decorator
    def loadMask(self, filename: str):
        dicom_ext = ['.dcm', '.ima']
        nii_ext = ['.nii', '.gz']
        npy_ext = ['.npy']
        npz_ext = ['.npz']
        path = os.path.abspath(filename)
        _, ext = os.path.splitext(path)

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
            if self.dicomHeaderList:
                # align datasets
                #find out if the loaded dataset is 3D or 2D stack
                self.setSplash(True, 1, 3, "Performing alignment")
                is2DStack = True
                try:
                    # in a 3D dataset, the following is not defined
                    spacing = self.dicomHeaderList[0].SpacingBetweenSlices
                    print("Aligning 2D Stack")
                except:
                    is2DStack = False
                    print("Aligning 3D Stack")

                if is2DStack:
                    transform = calcTransform2DStack(None, self.dicomHeaderList, None, dicomInfo)
                else:
                    transform = calcTransform(None, self.dicomHeaderList, None, dicomInfo, False)
                mask = transform(dataset*1000) > 990
            else:
                # we cannot align the datasets
                mask = dataset.squeeze()

            self.setSplash(True, 2, 3, "Importing masks")
            load_mask_validate(name, mask)
            self.setSplash(False, 0, 0, "")
            return

    ########################################################################################
    ###
    ### Deep learning functions
    ###
    ########################################################################################

    def setModelProvider(self, modelProvider):
        self.model_provider = modelProvider
        self.dl_classifier = modelProvider.load_model('Classifier')

    def setAvailableClasses(self, classList):
        self.toolbox_window.set_available_classes(classList)

    @pyqtSlot(str)
    def changeClassification(self, newClass):
        self.classifications[int(self.curImage)] = newClass

    @pyqtSlot()
    @snapshotSaver
    @separate_thread_decorator
    def doSegmentation(self):
        # run the segmentation
        imIndex = int(self.curImage)
        class_str = self.classifications[imIndex]
        self.setSplash(True, 0, 3, "Loading model...")
        try:
            segmenter = self.dl_segmenters[class_str]
        except KeyError:
            segmenter = self.model_provider.load_model(class_str)
            self.dl_segmenters[class_str] = segmenter

        self.setSplash(True, 1, 3, "Running segmentation...")
        t = time.time()
        inputData = {'image': self.imList[imIndex], 'resolution': self.resolution[0:2]}
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
