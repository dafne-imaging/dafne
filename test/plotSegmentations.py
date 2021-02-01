#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plotSegmentations(ima, segmentations):
    for label, mask in segmentations.items():
        plt.figure()
        imaRGB = np.stack([ima, ima, ima], axis = -1)
        imaRGB = imaRGB / imaRGB.max() * 0.6
        imaRGB[:,:,0] = imaRGB[:,:,0] + 0.4 * mask
        plt.imshow(imaRGB)
        plt.title(label)