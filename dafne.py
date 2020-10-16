#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:10:41 2015

@author: francesco
"""

#import sip
#sip.setapi('QString', 1)

import matplotlib
import argparse
matplotlib.use("Qt5Agg")

from ui.MuscleSegmentation import MuscleSegmentation

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
    
    
