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

import math
import numpy as np
import scipy.ndimage as ndimage

try:
    import SimpleITK as sitk # needs the SimpleElastix package!
except:
    sitk = None
        
ERROR_TOL = 1e-4

from ..dl.common.padorcut import padorcut

# code by Satya Mallick from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n >= ERROR_TOL:
        print("Warning! Rotation Matrix not unitary! Norm", n)
    return n < ERROR_TOL
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    #assert(isRotationMatrix(R))
    isRotationMatrix(R)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < ERROR_TOL
 
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.rad2deg(np.array([x, y, z]))


class DatasetTransform:
    def __init__(self):
        self.swapSlicesMoving = False # This is referring to the moving image (e.g. the mask)
        self.swapSlicesFix = False # This is referring to the fixed image 
        self.rotationAngles = None
        self.shift = None
        self.zoom = None
        self.endSize = None
        self.elastixTransform = []
        
    def isValid(self):
        return (self.rotationAngles is not None and
                self.shift is not None and
                self.zoom is not None and
                self.endSize is not None)
        
    def __getstate__(self):
        curTransformList = []
        for transform in self.elastixTransform:
            curTransformList.append(transform.asdict())
        
        outputDict = { 'swapSlicesFix': self.swapSlicesFix,
                      'swapSlicesMoving': self.swapSlicesMoving,
                      'rotationAngles': self.rotationAngles,
                      'shift': self.shift,
                      'zoom': self.zoom,
                      'endSize': self.endSize,
                      'elastixTransform': curTransformList
                      }
        return outputDict
    
    def __setstate__(self, state):
        self.swapSlicesFix = state['swapSlicesFix']
        self.swapSlicesMoving = state['swapSlicesMoving']
        self.rotationAngles = state['rotationAngles']
        self.shift = state['shift']
        self.zoom = state['zoom']
        self.endSize = state['endSize']
        
        self.elastixTransform = []
        for transform in state['elastixTransform']:
            self.elastixTransform.append(sitk.ParameterMap(transform))
            
    def transform(self, datasetIn, mode2d=False, maskMode=False):
        # the moving image (e.g. the mask) needs to have the slices flipped to be consistent
        if self.swapSlicesMoving:
            dataset = datasetIn[:,:,::-1]
        else:
            dataset = datasetIn

        if maskMode:
            interp_order = 0
        else:
            interp_order = 3

        #print "Rotating 1"
        dataMovRotated = ndimage.rotate(dataset, self.rotationAngles[0], axes=(1,2), order=interp_order)
        #print "Rotating 2"
        dataMovRotated = ndimage.rotate(dataMovRotated, self.rotationAngles[1], axes=(0,2), order=interp_order)
        #print "Rotating 3"
        dataMovRotated = ndimage.rotate(dataMovRotated, self.rotationAngles[2], axes=(0,1), order=interp_order)
        if not mode2d:
            dataMovExtended = ndimage.zoom(dataMovRotated, self.zoom, order=interp_order)
            dataMovShifted = ndimage.shift(dataMovExtended, self.shift, order=interp_order)
        else:
            #print("2D mode")
            dataMovExtended = ndimage.zoom(dataMovRotated, (self.zoom[0], self.zoom[1], 1), order=interp_order )
            dataMovShifted = ndimage.shift(dataMovExtended, (self.shift[0], self.shift[1], self.shift[2]/self.zoom[2]), order=interp_order)
            dataMovShifted = ndimage.zoom(dataMovShifted, (1, 1, self.zoom[2]), order = 0 )
        
        #dataMov2 = padorcut(dataMovShifted, self.endSize)
        dataMov2 = padorcut(dataMovShifted, np.array(self.endSize))
        if self.elastixTransform:
            # apply transformix
            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.SetLogToConsole(False)
            transformixImageFilter.SetLogToFile(False)
            if maskMode:
                for t in self.elastixTransform:
                    t['ResampleInterpolator'] = ["FinalNearestNeighborInterpolator"]
            transformixImageFilter.SetTransformParameterMap(self.elastixTransform)
            transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(dataMov2))
            transformixImageFilter.Execute()
            datasetOut = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
            
        else:
            datasetOut = dataMov2
        #datasetOut = padorcut(datasetOut, self.endSize)
        
        # the original fixed image (i.e. the target of the alignment has the slices swapped, so swap back)
        if self.swapSlicesFix:
            datasetOut = datasetOut[:,:,::-1]
        return datasetOut

    def __call__(self, datasetIn, mode2d = False, maskMode = False):
        return self.transform(datasetIn, mode2d, maskMode)


#NSLICES_MOVING = 47
#
#print "Loading data 1"
#dataMov, dataMovInfo = load3dDicom(DATA_MOVING)
#print "Loading data 2"
#dataFix, dataFixInfo = load3dDicom(DATA_FIX)
#
#dataFix = dataFix[:,:,0:NSLICES_MOVING]
#dataFixInfo = dataFixInfo[0:NSLICES_MOVING]

def findSeparateSlices(dicomInfoArray):
    separateIndices = []
    separateSlicePos = []
    for index, slcInfo in enumerate(dicomInfoArray):
        pos = slcInfo.ImagePositionPatient
        if pos not in separateSlicePos:
            separateSlicePos.append(pos)
            separateIndices.append(index)
    return separateIndices

class TimeSeriesTransform:
    def __init__(self):
        self.transformList = []
        
    def _transformSingle(self, dataset3D, transform):
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transform)
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(dataset3D))
        transformixImageFilter.Execute()
        return sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())
        
    def transform(self, dataset3D, timepoint = None):
        if timepoint is not None:
            return self._transformSingle(dataset3D, self.transformList[timepoint])
        
        datasetOutList = []
        datasetOutList.append(dataset3D)
        for transform in self.transformList:
            datasetOutList.append( self._transformSingle(dataset3D, transform) )
            
        return np.stack(datasetOutList, axis=3)
    
    def __getstate__(self):
        transformListSave = []
        for transformForSlice in self.transformList:
            transformsLocal = []
            for transform in transformForSlice:
                transformsLocal.append(transform.asdict())
            transformListSave.append(transformsLocal)
        return transformListSave
    
    def __setstate__(self, state):
        self.transformList = []
        for transformForSlice in state:
            transformsLocal = []
            for transformDict in transformForSlice:
                transformsLocal.append( sitk.ParameterMap(transformDict) )
            self.transformList.append(transformsLocal)
    
    def append(self, obj):
        self.transformList.append(obj)
    

def calcTimeseriesTransform(data4D):
    transformList = TimeSeriesTransform()
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(data4D[:,:,:,0]))
    for i in range(1,data4D.shape[3]):
        print("Registering %d of %d" % (i, data4D.shape[3]-1))
        elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(data4D[:,:,:,i]))
        elastixImageFilter.Execute()
        transformList.append(elastixImageFilter.GetTransformParameterMap())
        
    return transformList

# DataMov must be 3D because we need to rotate it
def calcTransform(dataFix, dataFixInfo, dataMov, dataMovInfo, doElastix = True):
        
    if type(dataFixInfo) != list:
        dataFixInfo = [dataFixInfo] # make sure that dataFixInfo is a list
        
    
    twoDMode = (len(dataFixInfo) == 1) # this identifies 2d mode: fix data is a single slice
    assert not twoDMode or not doElastix, "Elastix registration can only be used in 3D mode"    
    
    #print("2D Mode", twoDMode)
    
    transform = DatasetTransform()
    
    movSz = np.array([dataMovInfo[0].Rows, dataMovInfo[0].Columns, len(dataMovInfo)], dtype = np.uint16)
    fixSz = np.array([dataFixInfo[0].Rows, dataFixInfo[0].Columns, len(dataFixInfo)], dtype = np.uint16)
    
    # print(movSz, fixSz)
    
    rowMovOrientation = np.array(dataMovInfo[0].ImageOrientationPatient[3:6])
    colMovOrientation = np.array(dataMovInfo[0].ImageOrientationPatient[0:3])
    slcMovOrientation = np.cross(rowMovOrientation, colMovOrientation)
    
    # get difference between two slices
    sliceMovDiff = np.array(dataMovInfo[1].ImagePositionPatient) - np.array(dataMovInfo[0].ImagePositionPatient)
    
    if np.dot(sliceMovDiff, slcMovOrientation) < 0:
        transform.swapSlicesMoving = True
        if doElastix:
            dataMov = dataMov[:,:,::-1] # invert slice orientation
        dataMovInfo = dataMovInfo[::-1]
    
    rowFixOrientation = np.array(dataFixInfo[0].ImageOrientationPatient[3:6])
    colFixOrientation = np.array(dataFixInfo[0].ImageOrientationPatient[0:3])
    slcFixOrientation = np.cross(rowFixOrientation, colFixOrientation)
    
    if not twoDMode:
        # get difference between two slices
        sliceFixDiff = np.array(dataFixInfo[1].ImagePositionPatient) - np.array(dataFixInfo[0].ImagePositionPatient)
        
        if np.dot(sliceFixDiff, slcFixOrientation) < 0:
            transform.swapSlicesFix = True
            if doElastix:
                dataFix = dataFix[:,:,::-1] # invert slice orientation
            dataFixInfo = dataFixInfo[::-1]
        
    # print(np.array(dataMovInfo[0].ImagePositionPatient))
    # print(np.array(dataFixInfo[0].ImagePositionPatient))
    
    ijkMov = np.array([rowMovOrientation, colMovOrientation, slcMovOrientation]).T
    ijkFix = np.array([rowFixOrientation, colFixOrientation, slcFixOrientation]).T
    
    # print(ijkMov, ijkFix)
    
    MovToFix = np.matmul(ijkFix.T, ijkMov)
    
    angles = rotationMatrixToEulerAngles(MovToFix)
    
    transform.rotationAngles = angles
    
    try:
        sliceResMov = dataMovInfo[0].SpacingBetweenSlices
    except:
        sliceResMov = dataMovInfo[0].SliceThickness
        
    try:
        sliceResFix = dataFixInfo[0].SpacingBetweenSlices
    except:
        sliceResFix = dataFixInfo[0].SliceThickness
    
    if twoDMode: sliceResFix = dataFixInfo[0].SliceThickness # force slice thickness in case of 2d mode
    
    
    MovResolution = np.hstack([ dataMovInfo[0].PixelSpacing, sliceResMov ])
    FixResolution = np.hstack([ dataFixInfo[0].PixelSpacing, sliceResFix ])
    
    MovCenterPoint = (movSz.astype(np.float32)-1)/2
    FixCenterPoint = (fixSz.astype(np.float32)-1)/2

    if doElastix:
        print("Rotating...")
        dataMovRotated = ndimage.rotate(dataMov, angles[0], axes=(1,2))
        dataMovRotated = ndimage.rotate(dataMovRotated, angles[1], axes=(0,2))
        dataMovRotated = ndimage.rotate(dataMovRotated, angles[2], axes=(0,1))
        
    rotatedMovResolution = np.matmul(MovToFix, MovResolution)
    
    zoom = np.abs(rotatedMovResolution/FixResolution)
    
    transform.zoom = zoom
    
    if doElastix:
        dataMovExtended = ndimage.zoom(dataMovRotated, zoom)
    
    # MovCenterPointWorld = np.matmul(ijkMov,MovCenterPoint.T*MovResolution + np.hstack([ dataMovInfo[0].PixelSpacing, dataMovInfo[0].SliceThickness ])/2 ) + dataMovInfo[0].ImagePositionPatient 
    # FixCenterPointWorld = np.matmul(ijkFix,FixCenterPoint.T*FixResolution + np.hstack([ dataFixInfo[0].PixelSpacing, dataFixInfo[0].SliceThickness ])/2 ) + dataFixInfo[0].ImagePositionPatient
    MovCenterPointWorld = np.matmul(ijkMov,MovCenterPoint.T*MovResolution ) + dataMovInfo[0].ImagePositionPatient 
    FixCenterPointWorld = np.matmul(ijkFix,FixCenterPoint.T*FixResolution ) + dataFixInfo[0].ImagePositionPatient
    
    #print(MovCenterPointWorld, FixCenterPointWorld)
    
    diffCenterWorld = (FixCenterPointWorld-MovCenterPointWorld)
    
    diffCenter = np.matmul(ijkFix.T, diffCenterWorld.T)/FixResolution
    
    shift = -diffCenter
    
    transform.shift = shift
    
    transform.endSize = fixSz
    
    if doElastix:
        dataMovShifted = ndimage.shift(dataMovExtended, shift)
        
        #dataFix2 = padorcut(dataFix, fixSz + 8)
        dataMov2 = padorcut(dataMovShifted, fixSz)
        
        #fixMask = np.zeros_like(dataFix2)
        #fixMask[dataFix2 > 1] = 1
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetLogToFile(False)
        
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(dataFix))
        #elastixImageFilter.SetFixedMask(sitk.GetImageFromArray(fixMask.astype(np.uint8)))
        elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(dataMov2))
        print("Registering...")
        #elastixImageFilter.SetLogToFile(True)
        #elastixImageFilter.SetLogFileName('elastix.log')
        elastixImageFilter.Execute()
        
        transform.elastixTransform = elastixImageFilter.GetTransformParameterMap()
        print("Done")
    return transform
    
class DatasetTransform2D:
    def __init__(self):
        self.transformList = []
    
    def transform(self, datasetIn, maskMode = False):
        datasetOut = []
        for index, datasetTransform in enumerate(self.transformList):
            print(f"Aligining slice {index} of {len(self.transformList)}")
            datasetOut.append(datasetTransform(datasetIn, True, maskMode))
        return np.concatenate(datasetOut, axis = 2)
    
    def __call__(self, datasetIn, maskMode = False):
        return self.transform(datasetIn, maskMode)
    
def calcTransform2DStack(dataFix, dataFixInfo, dataMov, dataMovInfo):
    transform2D = DatasetTransform2D()
    for slc in dataFixInfo:
        transform2D.transformList.append(calcTransform(None, slc, None, dataMovInfo, False))
    return transform2D