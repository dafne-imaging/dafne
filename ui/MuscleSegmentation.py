#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:14:33 2020

@author: francesco
"""
import matplotlib

matplotlib.use("Qt5Agg")

import sys, os, time

# print(sys.path)
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(SCRIPT_DIR))

from ui.pySplineInterp import SplineInterpROIClass
from ui.ToolboxWindow import ToolboxWindow
from .pyDicomView import ImageShow
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import shutil
from datetime import datetime

from matplotlib.patches import Circle
import numpy as np
import scipy.ndimage as ndimage
import pickle
import os.path

# import SimpleITK as sitk
import re

import subprocess

from utils.dicomUtils import load3dDicom, save3dDicom

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
ROT_OTHER_COLOR = ROI_OTHER_COLOR_ORIG

ROI_FILENAME = 'rois.p'
AUTOSAVE_INTERVAL = 30

HIDE_ROIS_RIGHTCLICK = True

COLORS = ['blue', 'red', 'green', 'yellow', 'magenta', 'cyan', 'indigo', 'white', 'grey']


# define a circle with a contains method that for some reason does not work with conventional circles
class MyCircle(Circle):
    def __init__(self, xy, *args, **kwargs):
        Circle.__init__(self, xy, *args, **kwargs)
        self.xy = xy

    def contains(self, event):
        return ((event.xdata - self.xy[0]) ** 2 + (event.ydata - self.xy[1]) ** 2) < self.get_radius() ** 2


class MuscleSegmentation(ImageShow, QObject):
    def __init__(self, *args, **kwargs):
        ImageShow.__init__(self, *args, **kwargs)
        QObject.__init__(self)
        self.fig.canvas.mpl_connect('close_event', self.closeCB)
        #self.instructions = "Shift+click: add point, Shift+dblclick: optimize/simplify, Ctrl+click: remove point, Ctrl+dblclick: delete ROI, n: propagate fw, b: propagate back"
        self.setupToolbar()
        self.roiStack = None
        self.transforms = {}
        self.invtransforms = {}
        self.allROIs = {}  # allROIs is dict[roiName: (subroi)list[dict[imageNumber: Splines]]]
        self.wacom = False
        self.roiColor = ROI_COLOR
        self.roiOther = ROT_OTHER_COLOR
        self.roiSame = ROI_SAME_COLOR
        self.saveDicom = False

        self.model_provider = None
        self.dl_classifier = None
        self.dl_segmenters = {}
        self.classifications = []

        # self.fig.canvas.setCursor(Qt.BlankCursor)
        self.app = None

        # self.setCmap('viridis')
        self.extraOutputParams = []
        self.transformsChanged = False

        roi_bak_name = ROI_FILENAME + '.' + datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            shutil.copyfile(ROI_FILENAME, roi_bak_name)
        except:
            print("Warning: cannot copy roi file")

        self.lastsave = datetime.now()
        self.hideRois = False

    def setupToolbar(self):
        self.toolbox_window = ToolboxWindow()
        self.toolbox_window.show()

        self.toolbox_window.roi_added.connect(self.addRoi)
        self.toolbox_window.subroi_added.connect(self.addSubRoi)

        self.toolbox_window.roi_deleted.connect(self.removeRoi)
        self.toolbox_window.subroi_deleted.connect(self.removeSubRoi)

        self.toolbox_window.roi_changed.connect(self.changeRoi)

        self.toolbox_window.roi_clear.connect(self.clearCurrentROI)

        self.toolbox_window.autosegment_triggered.connect(self.doSegmentation)

        # tb = self.fig.canvas.toolbar
        # tb.addSeparator()
        #
        # segmentAction = tb.addAction("Muscle:")
        # segmentAction.setEnabled(False)
        #
        # self.setState('INACTIVE')
        #
        # self.combo = QComboBox()
        # self.combo.setEditable(True)
        # self.combo.resize(150,10)
        # tb.addWidget(self.combo)
        # self.combo.currentIndexChanged.connect(self.comboChange)
        #
        # tb.addAction("Delete", self.deleteComboItem)
        #
        # tb.addAction("Optimize", self.optimize)
        # tb.addAction("Simplify", self.simplify)
        #
        # self.calcTransformsAction = tb.addAction("Calculate transforms", self.calcTransforms)
        #
        # self.propagateAction = tb.addAction("Propagate forward", self.propagate)
        # #self.propagateAllAction =tb.addAction("Propagate fw all", self.propagateAll)
        #
        # tb.addSeparator()
        #
        # self.propagateAction = tb.addAction("Propagate back", self.propagateBack)
        # #self.propagateAction = tb.addAction("Propagate back all", self.propagateBackAll)
        # tb.addSeparator()
        #
        # tb.addAction("Save masks", self.saveResults)
        #
        # tb.addSeparator()
        # self.wacomAction = tb.addAction("Wacom mode", self.toggleWacom)
        # self.wacomAction.setCheckable(True)

    # def toggleWacom(self, wacomState = None):
    #     if wacomState is not None: self.wacom = not wacomState # force a toggle
    #     if self.wacom:
    #         self.wacom = False
    #         self.roiColor = ROI_COLOR_ORIG
    #         self.roiOther = ROI_OTHER_COLOR_ORIG
    #     else:
    #         self.wacom = True
    #         self.roiColor = ROI_COLOR_WACOM
    #         self.roiOther = ROI_OTHER_COLOR_WACOM
    #     self.wacomAction.setChecked(self.wacom)
    #     #self.refreshCB()
    #     self.redraw()

    @pyqtSlot(str)
    def removeRoi(self, roi_name):
        print("RemoveRoi")
        print(self.allROIs)
        for roiStack in self.allROIs[roi_name]:
            for k, roi in roiStack.items():
                roi.remove()
        del self.allROIs[roi_name]
        self.toolbox_window.set_rois_list(self.allROIs)
        self.refreshCB()

    @pyqtSlot(int)
    def removeSubRoi(self, subroi_number):
        current_name, _ = self.toolbox_window.get_current_roi_subroi()
        for _, roi in self.allROIs[current_name][subroi_number].items():
            roi.remove()

        self.allROIs[current_name].pop(subroi_number)
        if not self.allROIs[current_name]:
            self.roiStack = {}
            self.allROIs[current_name].append(
                self.roiStack)  # cannot have zero subrois. If we removed the last one, readd it
        self.toolbox_window.set_rois_list(self.allROIs)
        self.refreshCB()

    @pyqtSlot(str)
    def addRoi(self, roiName):
        try:
            self.roiStack = self.allROIs[roiName][0]
        except KeyError:
            self.roiStack = {}
            self.allROIs[roiName] = []
            self.allROIs[roiName].append(self.roiStack)
            self.setState("MUSCLE")
            self.toolbox_window.set_rois_list(self.allROIs)

        self.toolbox_window.set_current_roi(roiName, 0)
        self.refreshCB()

    @pyqtSlot()
    def addSubRoi(self):
        roi_name, _ = self.toolbox_window.get_current_roi_subroi()
        print(roi_name)
        self.roiStack = {}
        self.allROIs[roi_name].append(self.roiStack)
        self.toolbox_window.set_rois_list(self.allROIs)
        self.toolbox_window.set_current_roi(roi_name, len(self.allROIs[roi_name]) - 1)
        self.refreshCB()

    @pyqtSlot(str, int)
    def changeRoi(self, roi_name, subroi_index):
        print(roi_name, subroi_index)
        self.roiStack = self.allROIs[roi_name][subroi_index]
        self.refreshCB()

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
        return elastixImageFilter.GetTransformParameterMap()

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

    def simplify(self):
        r = self.getCurrentROI()
        # self.setCurrentROI(r.getSimplifiedSpline(SIMPLIFIED_ROI_POINTS))
        # self.setCurrentROI(r.getSimplifiedSpline(spacing=SIMPLIFIED_ROI_SPACING))
        self.setCurrentROI(r.getSimplifiedSpline3())
        # self.refreshCB()
        self.redraw()

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
        # self.refreshCB()
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

    def runTransformix(self, knots, transform):
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

        return knotsOut

    def propagate(self):
        if self.curImage >= len(self.imList) - 1: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]
        curROI = self.getCurrentROI()
        nextROI = self.getCurrentROI(+1)

        qbar = QProgressBar()
        qbar.setRange(0, 3)
        qbar.setWindowTitle(QString("Propagating"))
        qbar.setWindowModality(Qt.ApplicationModal)
        qbar.move(800, 500)
        qbar.show()

        qbar.setValue(0)
        plt.pause(.000001)

        knotsOut = self.runTransformix(curROI.knots, self.getTransform(int(self.curImage)))

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

        self.curImage += 1
        self.displayImage(self.imList[int(self.curImage)], self.cmap)
        self.redraw()

        qbar.setValue(1)
        plt.pause(.000001)

        self.simplify()

        qbar.setValue(2)
        plt.pause(.000001)
        self.optimize()
        qbar.close()

    def propagateBack(self):
        if self.curImage < 1: return
        # fixedImage = self.image
        # movingImage = self.imList[int(self.curImage+1)]
        curROI = self.getCurrentROI()
        nextROI = self.getCurrentROI(-1)

        qbar = QProgressBar()
        qbar.setRange(0, 3)
        qbar.setWindowTitle(QString("Propagating"))
        qbar.setWindowModality(Qt.ApplicationModal)
        qbar.move(800, 500)
        qbar.show()

        qbar.setValue(0)
        plt.pause(.000001)

        knotsOut = self.runTransformix(curROI.knots, self.getInverseTransform(int(self.curImage)))

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

        qbar.setValue(1)
        plt.pause(.000001)

        self.curImage -= 1
        self.displayImage(self.imList[int(self.curImage)], self.cmap)
        self.redraw()

        qbar.setValue(2)
        plt.pause(.000001)

        self.simplify()

        qbar.setValue(3)
        plt.pause(.000001)
        self.optimize()

        qbar.close()

    def setState(self, state):
        self.state = state

    def getState(self):
        if self.toolbox_window.valid_roi(): return 'MUSCLE'
        return 'INACTIVE'

    def addPoint(self, spline, event):
        self.currentPoint = spline.addKnot((event.xdata, event.ydata))
        # self.refreshCB()
        self.redraw()

    def leftPressCB(self, event):
        if not self.imPlot.contains(event):
            print("Event outside")
            return
        if self.getState() == 'MUSCLE':
            roi = self.getCurrentROI()

            knotIndex, knot = roi.findKnotEvent(event)
            print(self.toolbox_window.get_knot_button_state())
            print(ToolboxWindow.ADD_STATE)
            if self.toolbox_window.get_knot_button_state() == ToolboxWindow.REMOVE_STATE:
                if knotIndex is not None:
                    roi.removeKnot(knotIndex)
                    # self.refreshCB()
                    self.redraw()
            # if event.key == 'control' or event.key == 'super' or event.key == 'cmd':
            #     if event.dblclick:
            #         roi.removeAllKnots()
            #         self.redraw()
            #         return
            # elif event.key == 'shift' or checkCapsLock():
            elif self.toolbox_window.get_knot_button_state() == ToolboxWindow.ADD_STATE:
                if knotIndex is None:
                    self.addPoint(roi, event)
                else:
                    self.currentPoint = knotIndex

                #
                # if event.dblclick:
                #     self.optimize()
                #     self.simplify()
                #     self.redraw()

    @pyqtSlot()
    def clearCurrentROI(self):
        roi = self.getCurrentROI()
        roi.removeAllKnots()
        self.redraw()

    def movePoint(self, spline, event):
        if self.currentPoint is None:
            return

        spline.replaceKnot(self.currentPoint, (event.xdata, event.ydata))
        # self.refreshCB()
        self.redraw()

    def getCurrentROIName(self):
        return self.toolbox_window.get_current_roi_subroi()[0]

    def getCurrentSubroiNumber(self):
        return self.toolbox_window.get_current_roi_subroi()[1]

    def getCurrentROI(self, offset=0):
        if not self.getCurrentROIName():
            return None
        self.roiStack = self.allROIs[self.getCurrentROIName()][self.getCurrentSubroiNumber()]
        try:
            return self.roiStack[int(self.curImage + offset)]
        except KeyError:
            r = SplineInterpROIClass()
            self.roiStack[int(self.curImage + offset)] = r
            return r

    def setCurrentROI(self, r, offset=0):
        try:
            self.roiStack[int(self.curImage + offset)].remove()
        except:
            pass
        self.roiStack[int(self.curImage + offset)] = r

    def leftMoveCB(self, event):
        if self.getState() == 'MUSCLE':
            roi = self.getCurrentROI()
            if self.toolbox_window.get_knot_button_state() == ToolboxWindow.ADD_STATE:  # event.key == 'shift' or checkCapsLock():
                self.movePoint(roi, event)

    def leftReleaseCB(self, event):
        self.currentPoint = None  # reset the state

    def rightPressCB(self, event):
        self.hideRois = HIDE_ROIS_RIGHTCLICK
        self.redraw()

    def rightReleaseCB(self, event):
        self.hideRois = False
        self.redraw()

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
        # if not self.roiStack: return

        r = self.getCurrentROI()

        for name, subroisList in self.allROIs.items():
            for subroiNumber, stack in enumerate(subroisList):
                for sliceN, roi in stack.items():
                    if sliceN != int(self.curImage) or self.hideRois:
                        roi.remove()
                    else:
                        rSize = 0.1
                        rColor = self.roiOther
                        if name == self.getCurrentROIName():

                            if subroiNumber == self.getCurrentSubroiNumber():
                                rSize = ROI_CIRCLE_SIZE
                                rColor = self.roiColor
                            else:
                                rColor = self.roiSame
                        try:
                            roi.draw(self.axes, rSize, rColor)
                        except:
                            pass
        plt.draw()

    def closeCB(self, event):
        if not self.basepath: return
        self.toolbox_window.close()
        if self.transformsChanged: self.pickleTransforms()
        self.saveROIPickle()

    def saveROIPickle(self):
        roiPickleName = os.path.join(self.basepath, ROI_FILENAME)
        print("Saving ROIs", roiPickleName)
        if self.allROIs:  # make sure ROIs are not empty
            pickle.dump(self.allROIs, open(roiPickleName, 'wb'))

    def loadROIPickle(self):
        roiPickleName = os.path.join(self.basepath, ROI_FILENAME)
        print("Loading ROIs", roiPickleName)
        try:
            self.allROIs = pickle.load(open(roiPickleName, 'rb'))
        except UnicodeDecodeError:
            print('Warning: Unicode decode error')
            self.allROIs = pickle.load(open(roiPickleName, 'rb'), encoding='latin1')
        except:
            print("Unspecified error")
            return

        try:
            print(self.allROIs)
            assert type(self.allROIs[list(self.allROIs.keys())[0]][0]) == dict
        except:
            print("Unrecognized saved ROI type")
            self.allROIs = {}

        print('Rois loaded')
        print(self.allROIs)

        self.toolbox_window.set_rois_list(self.allROIs)

    def loadDirectory(self, path):
        ImageShow.loadDirectory(self, path)
        self.unPickleTransforms()
        self.loadROIPickle()
        self.refreshCB()

    def appendImage(self, im):
        ImageShow.appendImage(self, im)
        print("new Append Image")
        if not self.dl_classifier: return
        class_input = {'image': self.imList[-1], 'resolution': self.resolution[0:2]}
        class_str = self.dl_classifier(class_input)
        print("Classification", class_str)
        self.classifications.append(class_str)


    def saveResults(self):
        print("Saving results...")
        if not self.basepath: return
        roiBasePath = os.path.join(self.basepath, 'roi')
        imSize = self.image.shape

        if self.saveDicom:
            _, infoStack = load3dDicom(self.basepath)

        for roiName, subroiList in self.allROIs.items():
            masklist = []
            for imageIndex in range(len(self.imList)):
                roi = np.zeros(imSize)
                for subroiIndex in range(len(subroiList)):
                    try:
                        roi = np.logical_xor(roi, subroiList[subroiIndex][imageIndex].toMask(imSize, False))
                        # plt.figure()
                        # plt.imshow(roi)
                    except:
                        pass
                masklist.append(roi)
            print("Saving %s..." % (roiName))
            npMask = np.transpose(np.stack(masklist), [1, 2, 0])
            np.save("%s_%s.npy" % (roiBasePath, roiName), npMask)
            if self.saveDicom:
                save3dDicom(npMask, infoStack, os.path.join(self.basepath, "roi_ " + roiName))

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

    def keyPressCB(self, event):
        print(event.key)
        if 'shift' in event.key:
            self.toolbox_window.set_temp_knot_button_state(ToolboxWindow.ADD_STATE)
        elif 'control' in event.key or 'cmd' in event.key or 'super' in event.key or 'ctrl' in event.key:
            self.toolbox_window.set_temp_knot_button_state(ToolboxWindow.REMOVE_STATE)
        if event.key == 'n':
            self.propagate()
        elif event.key == 'b':
            self.propagateBack()
        else:
            ImageShow.keyPressCB(self, event)

    def keyReleaseCB(self, event):
        if 'shift' in event.key or 'control' in event.key or 'cmd' in event.key or 'super' in event.key or 'ctrl' in event.key:
            self.toolbox_window.restore_knot_button_state()

        # plt.show()

    ## Deep learning functions
    def setModelProvider(self, modelProvider):
        self.model_provider = modelProvider
        self.dl_classifier = modelProvider.load_model('Classifier')

    def setAvailableClasses(self, classList):
        self.toolbox_window.set_available_classes(classList)

    def displayImage(self, im, cmap = None):
        ImageShow.displayImage(self, im, cmap)
        self.toolbox_window.set_class(self.classifications[int(self.curImage)]) # update the classification combo

    @pyqtSlot(str)
    def changeClassification(self, newClass):
        self.classifications[int(self.curImage)] = newClass

    @pyqtSlot()
    def doSegmentation(self):
        # run the segmentation
        imIndex = int(self.curImage)
        class_str = self.classifications[imIndex]
        try:
            segmenter = self.dl_segmenters[class_str]
        except KeyError:
            segmenter = self.model_provider.load_model(class_str)
            self.dl_segmenters[class_str] = segmenter

        t = time.time()
        inputData = {'image': self.imList[imIndex], 'resolution': self.resolution[0:2]}
        print("Segmenting image...")
        masks_out = segmenter(inputData)
        print("Done")
        for name, mask in masks_out.items():
            splineInterpList = SplineInterpROIClass.FromMask(mask) # run mask tracing
            nContours = len(splineInterpList)
            if name not in self.allROIs:
                self.allROIs[name] = []
            nSubrois = len(self.allROIs[name])
            for contourIndex, contour in enumerate(splineInterpList):
                if contourIndex >= nSubrois:
                    self.allROIs[name].append({})
                self.allROIs[name][contourIndex][imIndex] = contour
        print("Segmentation time:", time.time() -t)
        self.toolbox_window.set_rois_list(self.allROIs)
        self.refreshCB()

