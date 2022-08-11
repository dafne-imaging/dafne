#!/usr/bin/env python3
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

import argparse


def main():
    parser = argparse.ArgumentParser(description='Batch model validation')
    parser.add_argument('--classification', type=str, help='Classification to use')
    parser.add_argument('--timestamp_start', type=int, help='Timestamp start')
    parser.add_argument('--timestamp_end', type=int, help='Timestamp end')
    parser.add_argument('--upload_stats', type=bool, help='Upload stats')
    parser.add_argument('--save_local', type=bool, help='Save local')
    parser.add_argument('--local_filename', type=str, help='Local filename')
    parser.add_argument('--roi', type=str, help='Reference ROI file')
    parser.add_argument('--masks', type=str, help='Reference Mask dataset')
    parser.add_argument('--comment', type=str, help='Comment for logging')
    parser.add_argument('dataset', type=str, help='Dataset for validation')

    args = parser.parse_args()
    args_to_pass = ['classification', 'timestamp_start', 'timestamp_end', 'upload_stats', 'save_local', 'local_filename']
    args_dict = {k: v for k,v in vars(args).items() if v is not None and k in args_to_pass}
    dataset = args.dataset
    roi = args.roi
    masks = args.masks

    from ..utils.BatchValidator import BatchValidator
    validator = BatchValidator(**args_dict)
    validator.load_directory(dataset)
    if roi:
        validator.loadROIPickle(roi)
    elif masks:
        validator.mask_import(masks)

    assert validator.mask_list, 'No masks found'
    validator.calculate(args.comment)

