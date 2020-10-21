#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:10:41 2015

@author: francesco
"""
from ui.MuscleSegmentation import MuscleSegmentation
from dl.LocalModelProvider import LocalModelProvider

import matplotlib
import argparse
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

MODELS_DIR = 'models'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Muscle segmentation tool.")
    parser.add_argument('path', type=str)
    parser.add_argument('-r', '--register', action='store_true', help='Perform the registration after loading.')
    parser.add_argument('-m', '--save-masks', action='store_true', help='Convert saved ROIs to masks.')
    parser.add_argument('-d', '--save-dicoms', action='store_true', help='Save ROIs as dicoms in addition to numpy')
    parser.add_argument('-w', '--wacom', action='store_true', help='Enable Wacom mode')
    parser.add_argument('-q', '--quit', action='store_true', help='Quit after loading the dataset (useful with -r or -q options).')
    
    args = parser.parse_args()
    
    
    imFig = MuscleSegmentation()
    #imFig.loadDirectory("image0001.dcm")


    dl_model_provider = LocalModelProvider(MODELS_DIR)
    available_models = dl_model_provider.available_models()

    available_models.remove('Classifier')
    imFig.setModelProvider(dl_model_provider)
    imFig.setAvailableClasses(available_models)

    imFig.loadDirectory(args.path)

    if args.save_dicoms:
        imFig.saveDicom = True
    
    if args.register:
        imFig.calcTransforms()
        imFig.pickleTransforms()
    
    if args.save_masks:
        imFig.saveResults()
    
    if args.wacom:
        print("Wacom mode")
        imFig.toggleWacom(True)
        
    
    if not args.quit:
        plt.show()
    
    
