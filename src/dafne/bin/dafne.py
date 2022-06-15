#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2022 Dafne-Imaging Team
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

import os
# Hide tensorflow warnings; set to 1 to see warnings
from ..utils import log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2', '3'}

from ..ui.MuscleSegmentation import MuscleSegmentation
from ..config import GlobalConfig, load_config

import matplotlib
matplotlib.use("Qt5Agg")
import argparse
import matplotlib.pyplot as plt

MODELS_DIR = 'models_old'


def main():
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

    if GlobalConfig['REDIRECT_OUTPUT']:
        import sys

        log.log_objects['stdout'] = log.LogStream(GlobalConfig['OUTPUT_LOG_FILE'], sys.stdout if GlobalConfig['ECHO_OUTPUT'] else None)
        log.log_objects['stderr'] = log.LogStream(GlobalConfig['ERROR_LOG_FILE'], sys.stderr if GlobalConfig['ECHO_OUTPUT'] else None)

        sys.stdout = log.log_objects['stdout']
        sys.stderr = log.log_objects['stderr']

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
    
    if args.save_masks:
        imFig.saveResults()
    
    if not args.quit:
        plt.show()
    
    
