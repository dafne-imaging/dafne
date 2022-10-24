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

from numpy import linalg
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib as mpl

from .polyToMask import polyToMask
import scipy.ndimage as ndimage
import time
#import similaritymeasures
import random


def polyToMaskFast(poly_verts, imageSize):
    x, y = np.meshgrid(np.arange(imageSize[1]), np.arange(0,imageSize[0]))
    x, y = x.flatten(), y.flatten()

    p = mpl.path.Path(poly_verts)

    points = np.vstack((x,y)).T

    grid = p.contains_points(points, radius=0)
    grid = grid.reshape((imageSize[0],imageSize[1]))

    return grid
    

KNOT_RADIUS = 2


#makes a list unique. From somewhere on the Internet
def uniquify(seq, idfun=None): 
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def knot_contains(knot, event, radius = KNOT_RADIUS):
    return ((event.xdata - knot[0]) ** 2 + (event.ydata - knot[1]) ** 2) < radius ** 2


class SplineInterpROIClass:
    def __init__(self, smooth = False):
        self.knots = []
        self.isCurveValid = False
        self.isMaskValid = False
        self.points = None
        self.pointSegments = None
        self.smooth = smooth
        self.mask = None
        self.isMaskFast = False
        self.isVisible = False
                      
    def invalidate_precalculations(self):
        self.isCurveValid = False
        self.isMaskValid = False

    # TODO: Remove this and only keep removeAllKnots?
    def clear(self):
        self.removeAllKnots()
        #self.invalidate_precalculations()
        #for i in range(0, self.knots):
        #    self.removeKnot(i)

    # returns the center of the ROI
    def getCenterOfMass(self):
        points = self.getCurve()
        if points is None:
            return None
        return np.mean(points, 0)

    def getBoundingBox(self, padding = 0):
        points = self.getCurve()
        minxy = np.min(points, 0) - padding
        maxxy = np.max(points, 0) + padding

        return minxy, maxxy

    def isPointNearSegment(self, point, segment, tolerance):
        isNear = lambda p1,p2: (abs(p1[0]-p2[0]) <= tolerance) and (abs(p1[1]-p2[1]) <= tolerance)
        for pt in segment:
            if isNear(point, pt):
                return True

    # return None if point is not close to path, otherwise return previous knot
    def isPointNearPath(self, point, tolerance = 2):
        self.getCurve() # make sure the paths are calculated
        if self.pointSegments is None:
            return None
        
        for i in range(0, len(self.pointSegments)):
            if self.isPointNearSegment(point, self.pointSegments[i], tolerance):
                return (i+1) % len(self.knots) # this is because segment 0 corresponds to points 1 to 2
            
            
    def addKnots(self, knotList, checkProximity = True):
        for k in knotList:
            self.addKnot(k, checkProximity)
            
    def addKnot(self,point, checkProximity = True, tolerance = 2):
        #check if the knot already exists
        existingIndex = self.findExistingKnot(point)
        if existingIndex is not None: return existingIndex
        # see if the knot should be added between two existing knots: this happens if we are close to the path
        if checkProximity:
            index = self.isPointNearPath(point, tolerance)
        else:
            index = None
        if index is None:
            newIndex = len(self.knots)
        else:
            newIndex = index+1
        self.invalidate_precalculations()
        self.knots.insert(newIndex,point)
        return newIndex # return the index of the new point

    def insertKnot(self, index, point):
        self.invalidate_precalculations()
        self.knots.insert(index, point)
    
    def findNearestKnot(self, knot):
        minDistSq = 1e6
        minKIndex = -1
        for kindex,k in enumerate(self.knots):
            dsq = (k[0]-knot[0])**2 + (k[1]-knot[1])**2
            if dsq < minDistSq:
                minDistSq = dsq
                minKIndex = kindex
        
        return minKIndex
    
    # gets a not considering a circular knot structure
    def getKnot(self, index):
        index = int(index)
        if not self.knots: return None
        if index < 0:
            return self.getKnot(len(self.knots)+index)
        if index >= len(self.knots):
            return self.getKnot(index-len(self.knots))
        else:
            return self.knots[index]
        
    def replaceKnot(self, index, point):
        self.invalidate_precalculations()
        self.knots[index] = point

    # moves a knot by a delta
    def moveKnot(self, index, delta):
        self.invalidate_precalculations()
        xy = self.knots[index]
        self.replaceKnot(index, (xy[0]+delta[0], xy[1]+delta[1]))

    def reduceKnots(self, n_knots):
        """
        Reduces the number of knots to n_knots by uniformly removing knots.
        """
        if n_knots >= len(self.knots):
            return

        self.invalidate_precalculations()
        self.knots = self.knots[::len(self.knots)//n_knots]
        if len(self.knots) > n_knots:
            knots_to_remove = len(self.knots) - n_knots
            new_knots = []
            remove_every = np.ceil(float(len(self.knots)) / knots_to_remove)
            for i,k in enumerate(self.knots):
                if i % remove_every != 0:
                    new_knots.append(k)

            self.knots = new_knots

        if len(self.knots) > n_knots:
            self.knots = self.knots[:n_knots]

    def rotateKnotList(self, shift):
        self.invalidate_precalculations()
        self.knots = self.knots[shift:] + self.knots[:shift]


    def removeAllKnots(self):
        # remove all the knots
        while len(self.knots) > 0:
            self.removeKnot(0)
    
    def findExistingKnot(self, knot, tolerance = 0.1):
        tol = tolerance**2
        for kindex, k in enumerate(self.knots):
            dist = (knot[0] - k[0])**2 + (knot[1] - k[1])**2
            if dist < tol: return kindex
        return None
    
    def removeKnot(self, index):
        self.invalidate_precalculations()
        self.knots.pop(index)

    def findKnotEvent(self, event, radius = KNOT_RADIUS):
        for knot_number, knot in enumerate(self.knots):
            if knot_contains(knot, event):
                return (knot_number, knot)
        return (None, None)
        
    #converts this spline to mask of a defined size. Note! At the moment this will not work properly if the contour touches the edges!
    def toMask(self, size = None, fast = True):
        if size is None:
            minxy, maxxy = self.getBoundingBox()
            size = (int(maxxy[1]), int(maxxy[0]))
        desiredSize = (size[1], size[0])
        #print "Mask valid:", self.isMaskValid
        if not self.isMaskValid or desiredSize != self.mask.shape or fast != self.isMaskFast:
            #t = time.time()
            if fast:
                self.mask = polyToMaskFast(self.getCurve(), size)
                self.isMaskFast = False
            else:
                self.mask = polyToMask(self.getCurve(), size)
                self.isMaskFast = True
            #t = time.time() - t
            #print "polyToMask runtime", t
            self.isMaskValid = True
        return self.mask
                
    def isPointInside(self, point, imageSize = None):
        if imageSize is not None:
            mask = self.toMask(imageSize)
        elif not self.isMaskValid: # imageSize is none, but mask is not valid
            minxy, maxxy = self.getBoundingBox()
            imageSize = (int(maxxy[1]), int(maxxy[0]))
            mask = self.toMask(imageSize)
        else: # imageSize is none, but mask is valid, so use that. Values outside will be considered 0 anyway
            mask = self.mask
        z = ndimage.map_coordinates(mask, [ [ point[1] ], [ point[0] ] ], order = 0, mode = 'constant')
        return z[0]
        
    def isValid(self):
        return len(self.knots) >= 4             
    
    def getCurve(self, shift_curve=False):
        # cannot make a closed curve with less than 4 knots
        if len(self.knots) < 4:
            self.points = None
            self.pointSegments = None
            return
        if self.isCurveValid:
            return self.points # return precalculated points if they are still valid
        xnew = []
        ynew = []            
        self.pointSegments = [] # all the points divided by segments. This contour is pixelated regardless of the smooth parameter, for speed
        for i in range(3,len(self.knots)):
            #print(i)
            xpart, ypart = self.getSplinePart([self.knots[i-3], self.knots[i-2], self.knots[i-1], self.knots[i]])
            xnew.extend(xpart)
            ynew.extend(ypart)
            self.pointSegments.append(uniquify(np.round(list(zip(xpart,ypart))), lambda x: (x[0],x[1])))
            
        #close the curve
        xpart, ypart = self.getSplinePart([self.knots[len(self.knots)-3], self.knots[len(self.knots)-2], self.knots[len(self.knots)-1], self.knots[0]])
        xnew.extend(xpart)
        ynew.extend(ypart)
        self.pointSegments.append(uniquify(np.round(list(zip(xpart,ypart))), lambda x: (x[0],x[1])))
        xpart, ypart = self.getSplinePart([self.knots[len(self.knots)-2],self.knots[len(self.knots)-1], self.knots[0], self.knots[1]])
        xnew.extend(xpart)
        ynew.extend(ypart)
        self.pointSegments.append(uniquify(np.round(list(zip(xpart,ypart))), lambda x: (x[0],x[1])))
        xpart, ypart = self.getSplinePart([self.knots[len(self.knots)-1], self.knots[0], self.knots[1], self.knots[2]])
        knot0_position = len(xnew)
        xnew.extend(xpart)
        ynew.extend(ypart)
        self.pointSegments.append(uniquify(np.round(list(zip(xpart,ypart))), lambda x: (x[0],x[1])))
        self.points = list(zip(xnew, ynew))
        if shift_curve:
            self.points = self.points[knot0_position:] + self.points[:knot0_position]
        if not self.smooth:
            # pixelate the result
            self.points = np.round(self.points)
            self.points = uniquify(self.points, lambda x: (x[0],x[1]))
        self.isCurveValid = True # unless nothing changes, this curve stays valid
        self.points = np.array(self.points)
        return self.points
    
    def calcDistance(self, otherSpline, shift=0, minimize_distance=False):
        """
        Calculates the distance between this spline and another spline.
        If minimize_distance is True, the distance is minimized by shifting the order of the knots of the other spline.
        The two splines must have the same number of knots

        :param otherSpline: the other spline
        :param shift: the shift of the order of the knots of the other spline
        :param minimize_distance: if True, the distance is minimized by shifting the order of the knots of the other spline

        :return: the distance (current or minimum) between the two splines
        """
        assert len(self.knots) == len(otherSpline.knots), "Splines must have the same number of knots"

        def knot_distance(knot1, knot2):
            return np.sqrt((knot1[0] - knot2[0])**2 + (knot1[1] - knot2[1])**2)

        if minimize_distance:
            min_distance = 10000
            min_shift = 0
            for shift in range(len(self.knots)):
                d = self.calcDistance(otherSpline, shift, minimize_distance=False)
                if d < min_distance:
                    min_distance = d
                    min_shift = shift
            return min_distance, min_shift

        # we are not trying to minimize the distance
        distance = 0
        for i in range(len(self.knots)):
            distance += knot_distance(self.knots[i], otherSpline.knots[(i+shift)%len(self.knots)])

        return distance,shift
        
        print("Similarity", sim)
        
        return sim
        
        minxy, maxxy = self.getBoundingBox()
        otherMinxy, otherMaxxy = otherSpline.getBoundingBox()
        imageSize = (int(max(maxxy[1], otherMaxxy[1])), int(max(maxxy[0], otherMaxxy[0])))
        m1 = self.toMask(imageSize)
        m2 = otherSpline.toMask(imageSize)
        return np.sum(np.logical_xor(m1,m2))


    def getSimplifiedSpline(self): # simplifies the spline based on useful knots
    
        def isNear(point, splinePart, tolerance):
            x = splinePart[0]
            y = splinePart[1]
            tolsq = tolerance**2
            for p in range(len(x)):
                if (point[0] - x[p])**2 + (point[1]-y[p])**2 <= tolsq: return True
            return False
        
        def nearestPoint(point, splinePart):
            x = splinePart[0]
            y = splinePart[1]
            dist = 1000
            nearest = None
            for p in range(len(x)):
                d = (point[0] - x[p])**2 + (point[1]-y[p])**2
                if d < dist:
                    dist = d
                    nearest = (x[p], y[p])
            return nearest

        newSpline = SplineInterpROIClass()
        #curve = self.getCurve()
        #newSpline.addKnots(curve)
        newSpline.addKnots(self.knots)
        optimizeFurther = True
        while optimizeFurther:
            optimizeFurther = False
            if len(newSpline.knots) < 5: return newSpline
            knotRange = list(range(len(newSpline.knots)))
            random.shuffle(knotRange)
            #for i,k in enumerate(newSpline.knots):
            for i in knotRange:
                k = newSpline.getKnot(i)
                newKnots = []
                # get the spline part missing the current knot
                newKnots.append(newSpline.getKnot(i-2))
                newKnots.append(newSpline.getKnot(i-1))
                newKnots.append(newSpline.getKnot(i+1))
                newKnots.append(newSpline.getKnot(i+2))
                spPart = newSpline.getSplinePart(newKnots)
                #nearest = nearestPoint(k, spPart)
                if isNear(k, spPart, 0.5):
                    #print("Removing knot", i)
                    optimizeFurther = True
                    newSpline.removeKnot(i)
                    break
        return newSpline
    
    # this is usually 4 knots
    def getSplinePart(self, knots):
        # knots need to be 3
        xpoints = [knot[0] for knot in knots]
        ypoints = [knot[1] for knot in knots]
        npoints = int(abs(ypoints[2] - ypoints[1]) + abs(xpoints[2] - xpoints[1]))
        try:
            tckp, u = splprep([xpoints, ypoints])
            #xx,yy = splev(np.linspace(0,1,100), tckp)
            xnew,ynew = splev(np.linspace(u[1],u[2],npoints), tckp) # produce 100 points between the first and second knot
        except ValueError:
            # fall back to linear interpolation between the two middle points
            k1 = knots[1]
            k2 = knots[2]
            xnew = np.linspace(k1[0], k2[0], npoints)
            ynew = np.linspace(k1[1], k2[1], npoints)

        return (xnew, ynew)

    def moveCenterTo(self, new_center):
        """
        Translates the whole ROI by moving all the knots. The new location is where the new center of mass will be
        """
        old_center_of_mass = self.getCenterOfMass()
        if old_center_of_mass is None: return
        deltaX = new_center[0] - old_center_of_mass[0]
        deltaY = new_center[1] - old_center_of_mass[1]
        for i, k in enumerate(self.knots[:]):
            self.replaceKnot(i, (k[0] + deltaX, k[1] + deltaY))

    def rotateBySinCos(self, angle_sin, angle_cos):
        """
        Rotates the contour around the center, given sine and cosine
        """
        center = self.getCenterOfMass()
        if center is None: return
        def rotateKnot(point):
            relX = point[0] - center[0]
            relY = point[1] - center[1]
            newX = angle_cos*relX - angle_sin*relY
            newY = angle_sin*relX + angle_cos*relY
            return (newX + center[0], newY + center[1])

        for i,k in enumerate(self.knots[:]):
            self.replaceKnot(i, rotateKnot(k))

    def rotateByAngle(self, angle):
        """
        Rotates the contour around the center by angle (in radians)
        """
        self.rotateBySinCos(np.sin(angle), np.cos(angle))

    def rotateByTwoPoints(self, startPoint, endPoint):
        """
        Rotates the contour by the angle given by the two given points and the center
        """
        center = self.getCenterOfMass()
        if center is None: return
        startVec = np.array([startPoint[0] - center[0], startPoint[1] - center[1]])
        endVec = np.array([endPoint[0] - center[0], endPoint[1] - center[1]])

        startVec /= linalg.norm(startVec)
        endVec /= linalg.norm(endVec)

        cosine = np.dot(startVec, endVec)
        sine = np.cross(startVec, endVec)
        self.rotateBySinCos(sine, cosine)

    def getOrientation(self, testPoint = None, center = None):
        """
        Returns the orientation vector between the center and a given point (or the first knot)
        with respect to the horizontal
        """

        if testPoint is None:
            testPoint = self.knots[0]

        if center is None:
            center = self.getCenterOfMass()
        if center is None: return None
        referenceVector = np.array([1, 0])
        testVector = np.array([testPoint[0] - center[0], testPoint[1] - center[1]])
        testVector /= linalg.norm(testVector)
        cosine = np.dot(referenceVector, testVector)
        sine = np.cross(referenceVector, testVector)
        return np.arctan2(sine, cosine)

    def reorientByAngle(self, angle):
        current_orientation = self.getOrientation()
        rotate_angle = angle - current_orientation
        self.rotateByAngle(rotate_angle)

    def copy(self):
        newSpline = SplineInterpROIClass()
        newSpline.addKnots(self.knots, checkProximity=False)
        return newSpline

    def __len__(self):
        return len(self.knots)

    # for pickling/unpickling, to avoid dumping all the internal representations
    def __getstate__(self):
        return self.knots
    
    def __setstate__(self, knotlist):
        self.__init__()
        #self.addKnots(knotlist)
        self.knots = knotlist
