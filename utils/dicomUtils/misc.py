#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:05:23 2020

@author: francesco
"""

import pydicom as dicom
import numpy as np

def loadDicomFile(fname):
    ds = dicom.read_file(fname)
    # rescale dynamic range to 0-4095
    try:
        pixelData = ds.pixel_array.astype(np.float32)
    except:
        ds.decompress()
        pixelData = ds.pixel_array.astype(np.float32)
    ds.PixelData = ""
    return pixelData, ds