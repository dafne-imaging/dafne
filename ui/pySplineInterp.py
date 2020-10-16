# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:10:45 2015

@author: francesco
"""

from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from utils.polyToMask import polyToMask
import scipy.ndimage as ndimage
import time
#import similaritymeasures
import random
import potrace


def polyToMaskFast(poly_verts, imageSize):
    x, y = np.meshgrid(np.arange(imageSize[1]), np.arange(0,imageSize[0]))
    x, y = x.flatten(), y.flatten()

    p = mpl.path.Path(poly_verts)

    points = np.vstack((x,y)).T

    grid = p.contains_points(points, radius=0)
    grid = grid.reshape((imageSize[0],imageSize[1]))

    return grid
    

KNOT_RADIUS = 0.1

#makes a list unique. From somewhere on the Internet
def uniquify(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

class KnotRepresentation(Circle):
    def __init__(self, xy, radius = KNOT_RADIUS, color = 'blue'):
        self.xy = xy
        Circle.__init__(self, xy, radius, facecolor = 'none', edgecolor = color, linewidth = 1.0)
        
    def contains(self, event):
        return ((event.xdata-self.xy[0])**2 + (event.ydata-self.xy[1])**2) < self.get_radius()**2

class SplineInterpROIClass:
    def __init__(self, smooth = False):
        self.knots = []
        self.knotRepresentations = []
        self.isCurveValid = False
        self.isMaskValid = False
        self.points = None
        self.pointSegments = None
        self.smooth = smooth
        self.mask = None
        self.isMaskFast = False
                      
    def invalidate_precalculations(self):
        self.isCurveValid = False
        self.isMaskValid = False
        
    def clear(self):
        self.invalidate_precalculations()
        for i in range(0, self.knots):
            self.removeKnot(i)

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
            if (self.isPointNearSegment(point, self.pointSegments[i], tolerance)):
                return (i+1) % len(self.knots) # this is because segment 0 corresponds to points 1 to 2
            
            
    def addKnots(self, knotList, checkProximity = True):
        for k in knotList:
            self.addKnot(k, checkProximity)
            
    def addKnot(self,point, checkProximity = True):
        #check if the knot already exists
        existingIndex = self.findExistingKnot(point)
        if existingIndex is not None: return existingIndex
        # see if the knot should be added between two existing knots: this happens if we are close to the path
        if checkProximity:
            index = self.isPointNearPath(point)
        else:
            index = None
        if index is None:
            newIndex = len(self.knots)
        else:
            newIndex = index+1
        self.isCurveValid = False
        self.knots.insert(newIndex,point)
        self.knotRepresentations.insert(newIndex,KnotRepresentation(point))
        return newIndex # return the index of the new point
    
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
        try:
            self.knotRepresentations[index].remove()
        except:
            pass
        self.knotRepresentations[index] = KnotRepresentation(point)

    # moves a knot by a delta
    def moveKnot(self, index, delta):
        self.invalidate_precalculations()
        xy = self.knots[index]
        self.replaceKnot(index, (xy[0]+delta[0], xy[1]+delta[1]))
    
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
        try:
            self.knotRepresentations[index].remove()
        except:
            pass
        self.knotRepresentations.pop(index)

    def findKnotEvent(self, event):
        for i in range(0, len(self.knotRepresentations)):
            if self.knotRepresentations[i].contains(event):
                return (i,self.knots[i]) # returns the index of the knot and the knot itself if it contains the event
        return (None, None)
        
    def remove(self):
        for k in self.knotRepresentations:
            try:
                k.remove()
            except:
                pass
        try:
            self.plot.remove()
        except:
            pass
    
    def draw(self, axes, radius = KNOT_RADIUS, color = 'blue'):
        
        self.remove()
        
        for k in self.knotRepresentations:
            k.set_radius(radius)
            k.set_edgecolor(color)
            axes.add_patch(k)
        try:            
            points = self.getCurve()
            self.plot = Polygon(points, facecolor = 'none', edgecolor = color)
            axes.add_patch(self.plot)
        except:
            pass
            
        plt.draw()
        
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
    
    def getCurve(self):
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
        xnew.extend(xpart)
        ynew.extend(ypart)
        self.pointSegments.append(uniquify(np.round(list(zip(xpart,ypart))), lambda x: (x[0],x[1])))
        self.points = list(zip(xnew, ynew))
        if not self.smooth:
            # pixelate the result
            self.points = np.round(self.points)
            self.points = uniquify(self.points, lambda x: (x[0],x[1]))
        self.isCurveValid = True # unless nothing changes, this curve stays valid
        self.points = np.array(self.points)
        return self.points
        
    # get a modified spline with a specified number of knots or a specified spacing between knots
    def getSimplifiedSpline(self, nPoints = None, spacing = None):
        assert(nPoints or spacing)
        simplifiedPoints = np.round(self.points)
        simplifiedPoints = uniquify(simplifiedPoints, lambda x: (x[0],x[1]))
        
        if nPoints:
            assert(nPoints >= 4)
            step = int(len(simplifiedPoints) / nPoints)
            if step < 1: step = 1
        else:
            step = spacing
            if step*4 > len(simplifiedPoints): step = int(len(simplifiedPoints)/4)
            
            
        simplifiedPoints = simplifiedPoints[0:(-step+1):step]
        newSpline = SplineInterpROIClass()
        for p in simplifiedPoints:
            newSpline.addKnot(p)
        return newSpline
    
    def calcSimilarity(self, otherSpline):
        
        sim = similaritymeasures.area_between_two_curves(self.getCurve(), otherSpline.getCurve())
        
        print("Similarity", sim)
        
        return sim
        
        minxy, maxxy = self.getBoundingBox()
        otherMinxy, otherMaxxy = otherSpline.getBoundingBox()
        imageSize = (int(max(maxxy[1], otherMaxxy[1])), int(max(maxxy[0], otherMaxxy[0])))
        m1 = self.toMask(imageSize)
        m2 = otherSpline.toMask(imageSize)
        return np.sum(np.logical_xor(m1,m2))
    
    def getSimplifiedSpline2(self): # simplifies the spline based on useful knots
        t = time.time()
        sim_th = np.sum(self.toMask().astype(np.uint16))/100 # set threshold to 1% of area
        sim_th = min(sim_th, 20)
        print("Sim Threshold", sim_th)
        newSpline = SplineInterpROIClass()
        curve = self.getCurve()
        newSpline.addKnots(curve)
        #newSpline.addKnots(self.knots)
        optimizeFurther = True
        while optimizeFurther:
            optimizeFurther = False
            if len(newSpline.knots) < 5: return newSpline
            for i,k in enumerate(newSpline.knots):
                otherSpline = SplineInterpROIClass()
                otherKnots = newSpline.knots[:]
                otherKnots.pop(i)
                otherSpline.addKnots(otherKnots)
                if self.calcSimilarity(otherSpline) <= sim_th:
                    print("Removing knot", i)
                    optimizeFurther = True
                    newSpline.removeKnot(i)
                    break
        print("Simplifying time:", time.time() - t)
        return newSpline
    
    def getSimplifiedSpline3_slow(self): # simplifies the spline based on useful knots
        t = time.time()
        newSpline = SplineInterpROIClass()
        newSpline.addKnots(self.knots)
        optimizeFurther = True
        while optimizeFurther:
            optimizeFurther = False
            if len(newSpline.knots) < 5: return newSpline
            for i,k in enumerate(newSpline.knots):
                otherSpline = SplineInterpROIClass()
                otherKnots = newSpline.knots[:]
                otherKnots.pop(i)
                otherSpline.addKnots(otherKnots)
                if otherSpline.isPointNearPath(k, 1):
                    print("Removing knot", i)
                    optimizeFurther = True
                    newSpline.removeKnot(i)
                    break
        print("Simplifying time:", time.time() - t)
        return newSpline
            
    def getSimplifiedSpline3(self): # simplifies the spline based on useful knots
    
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

        t = time.time()
        newSpline = SplineInterpROIClass()
        curve = self.getCurve()
        newSpline.addKnots(curve)
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
        print("Simplifying time:", time.time() - t)
        return newSpline
    
    # this is usually 4 knots
    def getSplinePart(self, knots):
        # knots need to be 3
        xpoints = [knot[0] for knot in knots]
        ypoints = [knot[1] for knot in knots]
        tckp, u = splprep([xpoints, ypoints])
#        xx,yy = splev(np.linspace(0,1,100), tckp)
#        plt.plot(xx,yy)
        npoints = int(abs(ypoints[2] - ypoints[1]) + abs(xpoints[2] - xpoints[1]))
        #print("npoints", npoints)

        xnew,ynew = splev(np.linspace(u[1],u[2],npoints), tckp) # produce 100 points between the first and second knot
        return (xnew, ynew)
    
    # for pickling/unpickling, to avoid dumping all the internal representations
    def __getstate__(self):
        return self.knots
    
    def __setstate__(self, knotlist):
        self.__init__()
        self.addKnots(knotlist)
        
    @staticmethod
    def FromMask(maskImage):
        outputSplines = []
        bmp = potrace.Bitmap(maskImage)
        
        # Trace the bitmap to a path
        path = bmp.trace(alphamax = 1.33, opticurve = 1, opttolerance = 1)
        
        # Iterate over path curves
        for curve in path:
            contour = curve.tesselate(potrace.Curve.regular, 4)
            sp = SplineInterpROIClass()
            sp.addKnots(np.round(contour), False)
            outputSplines.append(sp)
        
        return outputSplines
        
            
if __name__ == "__main__":
    sp = SplineInterpROIClass()
    sp.addKnot((10,5))
    sp.addKnot((10,20))
    sp.addKnot((30,20))
    sp.addKnot((20,10))
    sp.addKnot((20,15))
    sp.addKnot((12,15))
    im = sp.toMask((35,40), False)
    plt.imshow(im)
    sp.draw(plt.gca(), color = 'red')    
    print(sp.isPointInside((10,6)))
    
    
    
    # # Create a bitmap from the array
    # bmp = potrace.Bitmap(im)
    
    # # Trace the bitmap to a path
    # path = bmp.trace(turnpolicy = potrace.TURNPOLICY_LEFT, alphamax = 1.33, opticurve = 1, opttolerance = 1)
    
    # # Iterate over path curves
    # for curve in path:
    #     print("start_point =", curve.start_point)
    #     for segment in curve:
    #         print(segment)
    #         end_point_x, end_point_y = segment.end_point
    #         if segment.is_corner:
    #             c_x, c_y = segment.c
    #         else:
    #             c1_x, c1_y = segment.c1
    #             c2_x, c2_y = segment.c2
                
    splines = SplineInterpROIClass.FromMask(im)
    splines[0].draw(plt.gca(), color = 'blue')
                
    plt.show()
