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

# new implementation, working on a generic number of axes
def padorcut(arrayin, newSize, axis = None):
    nDims = arrayin.ndim
    
    # extend dimensions
    while nDims < len(newSize):
        arrayin = np.expand_dims(arrayin, nDims)
        nDims = arrayin.ndim
        
    if type(axis) is int:
        # check if newsz is iterable, otherwise assume it's a number
        try:
            newSz = newSize[axis]
        except:
            newSz = int(newSize)
        oldSz = arrayin.shape[axis]
        if oldSz < newSz:
            padBefore = int(math.floor(float(newSz - oldSz)/2))
            padAfter = int(math.ceil(float(newSz - oldSz)/2))
            padding = []
            for i in range(nDims):
                if i == axis:
                    padding.append( (padBefore, padAfter) )
                else:
                    padding.append( (0,0) )
            return np.pad(arrayin, padding, 'constant')
        elif oldSz > newSz:
            cutBefore = int(math.floor(float(oldSz - newSz)/2))
            cutAfter = int(math.ceil(float(oldSz - newSz)/2))
            slc = [slice(None)]*nDims
            slc[axis] = slice(cutBefore, -cutAfter)
            return arrayin[tuple(slc)]
        else:
            return arrayin
    else:
        for ax in range(nDims):
            arrayin = padorcut(arrayin, newSize, ax)
        return arrayin

def translate(arrayin, translation):
    padBeforeX = translation[0] if translation[0] > 0 else 0
    padAfterX = -translation[0] if translation[0] < 0 else 0
    padBeforeY = translation[1] if translation[1] > 0 else 0
    padAfterY = -translation[1] if translation[1] < 0 else 0
    
    cutBeforeX = padAfterX
    cutAfterX = padBeforeX
    cutBeforeY = padAfterY
    cutAfterY = padBeforeY
    
    nx, ny = arrayin.shape    
    
    cutArray = arrayin[cutBeforeX:(nx-cutAfterX), cutBeforeY:(ny-cutAfterY)]
    arrayout = arrayout=np.pad(cutArray, ( (padBeforeX,padAfterX), (padBeforeY, padAfterY) ), 'constant')
    return arrayout
    
    