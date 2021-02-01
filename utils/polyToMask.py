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

import numpy as np
from collections import deque

# recursively flood a mask in place
def flood(seedX, seedY, mask):
    # check if we are on countour or out of bounds
    sz = mask.shape
    q = deque()
    if (mask[seedX][seedY] == 1):
        return
    q.append((seedX, seedY))
    
    # function to determine if a point is out of bound
    isOutOfBound = lambda p: p[0] < 0 or p[1] < 0 or p[0] >= sz[0] or p[1] >= sz[1]
    while q: # iterate until empty
        currentNode = q.popleft()
        if isOutOfBound(currentNode) or mask[currentNode[0]][currentNode[1]] == 1:
            continue
        # travel right (east)
        for e in range(currentNode[0], sz[0]):
            # exit if we reached a countour
            if mask[e][currentNode[1]] == 1:
                break
            # change color
            mask[e][currentNode[1]] = 1
            # add north and south to the queue
            q.append((e, currentNode[1]-1))
            q.append((e, currentNode[1]+1))
        # travel left (west)
        for w in range(currentNode[0]-1, -1, -1):
            # exit if we reached a countour
            if mask[w][currentNode[1]] == 1:
                break
            # change color
            mask[w][currentNode[1]] = 1
            # add north and south to the queue
            q.append((w, currentNode[1]-1))
            q.append((w, currentNode[1]+1))
            
    
# old recursive implementation that crashed python :(
#    
#    if seedX < 0 or seedX >= sz[0] or seedY < 0 or seedY >= sz[1] or mask[seedX][seedY] == 1:
#        return
#    mask[seedX][seedY] = 1
#    # recursively flood
#    flood(seedX+1, seedY, mask)
#    flood(seedX-1, seedY, mask)
#    flood(seedX, seedY+1, mask)
#    flood(seedX, seedY-1, mask)

def intround(x):
    return int(round(x))

#converts this spline to mask of a defined size. Note! At the moment this will not work properly if the contour touches the edges!
def polyToMask(points, size):
    size = (size[1], size[0]) # x is rows and y is columns
    mask = np.zeros((size[0]+1, size[1])) # create a mask that is 1 larger than needed
    for i in range(0, len(points)):
        curPoint = points[i]
        try:
            nextPoint = points[i+1]
        except IndexError:
            nextPoint = points[0] # close the polygon
            
        #print curPoint, nextPoint
        if (curPoint[0] == nextPoint[0]) and (curPoint[1] == nextPoint[1]):
            continue
        if curPoint[0] < 0: curPoint[0] = 0
        if curPoint[1] < 0: curPoint[1] = 0
        if nextPoint[0] < 0: nextPoint[0] = 0
        if nextPoint[1] < 0: nextPoint[1] = 0
        
        if curPoint[0] > size[0]-1: curPoint[0] = size[0]-1
        if curPoint[1] > size[1]-1: curPoint[1] = size[1]-1
        if nextPoint[0] > size[0]-1: nextPoint[0] = size[0]-1
        if nextPoint[1] > size[1]-1: nextPoint[1] = size[1]-1
            
        mask[intround(curPoint[0])][intround(curPoint[1])] = 1 # set initial point to 1
        # special case for vertical line
        if nextPoint[0] == curPoint[0]:
            # order start and end
            if curPoint[1] < nextPoint[1]:
                startY = curPoint[1]
                endY = nextPoint[1]
            else:
                startY = nextPoint[1]
                endY = curPoint[1]
            for y in range(intround(startY), intround(endY+1)): # how stupid is this?
                mask[intround(curPoint[0])][intround(y)] = 1
        else:
            # not a vertical line
            slope = (nextPoint[1]-curPoint[1])/(nextPoint[0]-curPoint[0])
            if abs(slope) < 1:
                #travel along x because line is "flat"
                if curPoint[0] < nextPoint[0]:
                    startX = curPoint[0]
                    endX = nextPoint[0]
                    startY = curPoint[1]
                else:
                    startX = nextPoint[0]
                    endX = curPoint[0]
                    startY = nextPoint[1]
                for x in range(intround(startX), intround(endX+1)):
                    nextY = startY + (x-startX)*slope
                    mask[intround(x)][intround(nextY)] = 1
            else:
                # travel along y because line is "steep"
                if curPoint[1] < nextPoint[1]:
                    startY = curPoint[1]
                    endY = nextPoint[1]
                    startX = curPoint[0]
                else:
                    startY = nextPoint[1]
                    endY = curPoint[1]
                    startX = nextPoint[0]
                for y in range(intround(startY), intround(endY+1)):
                    nextX = startX + (y-startY)/slope
                    mask[intround(nextX)][intround(y)] = 1
        
    contour = np.copy(mask) # save this for later
    flood(mask.shape[0]-1, 0, mask) # flood the outside of the contour, by starting on a point that is for sure outside
    # now the mask is actually the inverted mask and the contour
    # invert the mask and add the contour back
    mask = np.logical_or(contour, np.logical_not(mask))
    #mask = np.logical_not(mask) # do not add the contour! Change this for production, this is only for the abstract
    return np.transpose(mask[0:len(mask)-1]) # return the mask with the original desired size - transposed because x is rows and y is columns
           