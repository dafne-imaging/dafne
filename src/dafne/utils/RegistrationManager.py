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

import SimpleITK as sitk
import pickle, os, re
import numpy as np

TRANSFORMS_NAME = 'transforms.p'


class RegistrationManager:

    def __init__(self, image_list, transforms_filename=None, current_dir=None, temp_dir=None):
        self.image_list = image_list
        self.transforms_filename = transforms_filename
        self.current_dir = current_dir
        self.temp_dir = temp_dir

        self.transforms = {}
        self.invtransforms = {}
        if self.transforms_filename:
            self.unpickle_transforms()
        self.transforms_changed = False

    def set_standard_transforms_name(self, path, basename):
        if basename:
            self.transforms_filename = basename + '.' + TRANSFORMS_NAME
        else:
            self.transforms_filename = TRANSFORMS_NAME

        self.transforms_filename = os.path.join(path, self.transforms_filename)
        self.unpickle_transforms()

    def move_to_temp_dir(self):
        if not self.temp_dir or not self.current_dir: return
        os.chdir(self.temp_dir)

    def move_to_work_dir(self):
        if not self.temp_dir or not self.current_dir: return
        os.chdir(self.current_dir)

    def pickle_transforms(self):
        if not self.transforms_filename: return
        if not self.transforms_changed: return
        print("Pickling transforms", self.transforms_filename)
        pickle_obj = {}
        transform_dict = {}
        for k, transformList in self.transforms.items():
            cur_transform_list = []
            for transform in transformList:
                cur_transform_list.append(transform.asdict())
            transform_dict[k] = cur_transform_list
        inv_transform_dict = {}
        for k, transformList in self.invtransforms.items():
            cur_transform_list = []
            for transform in transformList:
                cur_transform_list.append(transform.asdict())
            inv_transform_dict[k] = cur_transform_list
        pickle_obj['direct'] = transform_dict
        pickle_obj['inverse'] = inv_transform_dict
        pickle.dump(pickle_obj, open(self.transforms_filename, 'wb'))

    def unpickle_transforms(self):
        print("Unpickling transforms", self.transforms_filename)
        if not self.transforms_filename: return
        try:
            pickle_obj = pickle.load(open(self.transforms_filename, 'rb'))
        except:
            print("Error trying to load transforms")
            return

        transform_dict = pickle_obj['direct']
        self.transforms = {}
        for k, transformList in transform_dict.items():
            cur_transform_list = []
            for transform in transformList:
                cur_transform_list.append(sitk.ParameterMap(transform))
            self.transforms[k] = tuple(cur_transform_list)
        inv_transform_dict = pickle_obj['inverse']
        self.invtransforms = {}
        for k, transformList in inv_transform_dict.items():
            cur_transform_list = []
            for transform in transformList:
                cur_transform_list.append(sitk.ParameterMap(transform))
            self.invtransforms[k] = tuple(cur_transform_list)

        print("Transforms", list(self.transforms.keys()))
        print("Inv Transforms", list(self.invtransforms.keys()))

    def get_inverse_transform(self, imIndex):
        try:
            return self.invtransforms[imIndex]
        except KeyError:
            self.calc_inverse_transform(imIndex)
            return self.invtransforms[imIndex]

    def get_transform(self, imIndex):
        try:
            return self.transforms[imIndex]
        except KeyError:
            self.calc_transform(imIndex)
            return self.transforms[imIndex]

    def calc_transform(self, imIndex):
        if imIndex >= len(self.image_list) - 1: return
        fixedImage = self.image_list[imIndex]
        movingImage = self.image_list[imIndex + 1]
        self.transforms[imIndex] = self.run_elastix(fixedImage, movingImage)
        self.transforms_changed = True

    def calc_inverse_transform(self, imIndex):
        if imIndex < 1: return
        fixedImage = self.image_list[imIndex]
        movingImage = self.image_list[imIndex - 1]
        self.invtransforms[imIndex] = self.run_elastix(fixedImage, movingImage)
        self.transforms_changed = True

    def run_elastix(self, fixedImage, movingImage):
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetLogToFile(False)

        elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(fixedImage))
        elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(movingImage))
        print("Registering...")

        self.move_to_temp_dir()
        elastixImageFilter.Execute()
        print("Done")
        pMap = elastixImageFilter.GetTransformParameterMap()
        self.clean_elastix_files()
        self.move_to_work_dir()
        return pMap

    def calc_transforms(self, callback_function=None):
        for imIndex in range(len(self.image_list)):
            print("Calculating image:", imIndex)
            # the transform was already calculated
            if imIndex not in self.transforms:
                self.calc_transform(imIndex)
            if imIndex not in self.invtransforms:
                self.calc_inverse_transform(imIndex)
            if callback_function is not None:
                callback_function(imIndex)

        print("Saving transforms")
        self.pickle_transforms()

    def clean_elastix_files(self):
        files_to_delete = ['TransformixPoints.txt',
                           'outputpoints.txt',
                           'TransformParameters.0.txt',
                           'TransformParameters.1.txt',
                           'TransformParameters.2.txt']

        for file in files_to_delete:
            try:
                os.remove(file)
            except:
                pass

    def run_transformix_mask(self, mask, transform):
        transformixImageFilter = sitk.TransformixImageFilter()

        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetLogToFile(False)

        for t in transform:
            t['ResampleInterpolator'] = ["FinalNearestNeighborInterpolator"]

        transformixImageFilter.SetTransformParameterMap(transform)

        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(mask))
        self.move_to_temp_dir()
        transformixImageFilter.Execute()

        mask_out = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        self.clean_elastix_files()
        self.move_to_work_dir()

        return mask_out.astype(np.uint8)

    def run_transformix_knots(self, knots, transform):
        transformixImageFilter = sitk.TransformixImageFilter()

        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetLogToFile(False)

        transformixImageFilter.SetTransformParameterMap(transform)

        self.move_to_temp_dir()

        # create Transformix point file
        with open("TransformixPoints.txt", "w") as f:
            f.write("point\n")
            f.write("%d\n" % (len(knots)))
            for k in knots:
                # f.write("%.3f %.3f\n" % (k[0], k[1]))
                f.write("%.3f %.3f\n" % (k[0], k[1]))

        transformixImageFilter.SetFixedPointSetFileName("TransformixPoints.txt")
        transformixImageFilter.SetOutputDirectory(".")
        transformixImageFilter.Execute()

        outputCoordRE = re.compile("OutputPoint\s*=\s*\[\s*([\d.]+)\s+([\d.]+)\s*\]")

        knotsOut = []

        with open("outputpoints.txt", "r") as f:
            for line in f:
                m = outputCoordRE.search(line)
                knot = (float(m.group(1)), float(m.group(2)))
                knotsOut.append(knot)

        self.clean_elastix_files()
        self.move_to_work_dir()

        return knotsOut
