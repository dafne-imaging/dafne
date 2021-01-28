#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:10:41 2015

@author: francesco
"""
import os
# Hide tensorflow warnings; set to 1 to see warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}

from ui.MuscleSegmentation import MuscleSegmentation
from dl.LocalModelProvider import LocalModelProvider
from dl.RemoteModelProvider import RemoteModelProvider
from config import GlobalConfig, load_config

import matplotlib
import argparse
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

MODELS_DIR = 'models_old'


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Muscle segmentation tool.")
    parser.add_argument('path', nargs='?', type=str)
    parser.add_argument('-c', '--class', dest='dl_class', type=str, help='Specify the deep learning model to use for the dataset')
    parser.add_argument('-r', '--register', action='store_true', help='Perform the registration after loading.')
    parser.add_argument('-m', '--save-masks', action='store_true', help='Convert saved ROIs to masks.')
    parser.add_argument('-d', '--save-dicoms', action='store_true', help='Save ROIs as dicoms in addition to numpy')
    parser.add_argument('-q', '--quit', action='store_true', help='Quit after loading the dataset (useful with -r or -q options).')
    parser.add_argument('-rm', '--remote-model', action='store_true', help='Force remote model')
    parser.add_argument('-lm', '--local-model', action='store_true', help='Force local model')
    
    args = parser.parse_args()

    load_config()

    if args.remote_model:
        GlobalConfig['MODEL_PROVIDER'] = 'Remote'

    if args.local_model:
        GlobalConfig['MODEL_PROVIDER'] = 'Local'

    imFig = MuscleSegmentation()

    dl_class = None

    if args.dl_class:
        dl_class = args.dl_class

    if args.path:
        imFig.loadDirectory(args.path, dl_class)

    if args.save_dicoms:
        imFig.saveDicom = True
    
    if args.register:
        imFig.calcTransforms()
        imFig.pickleTransforms()
    
    if args.save_masks:
        imFig.saveResults()
    
    if not args.quit:
        plt.show()
    
    
