#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dl.DynamicDLModel import DynamicDLModel
from utils.dicomUtils import loadDicomFile
#import numpy as np
from test.plotSegmentations import plotSegmentations
import matplotlib.pyplot as plt

def testSegmentation(modelPath, dicomPath):
    thighModel = DynamicDLModel.Load(open(modelPath, 'rb'))
    
    ima, info = loadDicomFile(dicomPath)
    
    resolution = info.PixelSpacing
    out = thighModel({'image': ima, 'resolution': resolution})
    
    plotSegmentations(ima, out)
    
    plt.show()
    
def testClassification(modelPath, dicomPath):
    classModel = DynamicDLModel.Load(open(modelPath, 'rb'))
    
    ima, info = loadDicomFile(dicomPath)
    
    resolution = info.PixelSpacing
    out = classModel({'image': ima, 'resolution': resolution})
    print(out)