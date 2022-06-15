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

def biascorrection(file_or_image):
    if type(file_or_image) == str:
        return biascorrection_file(file_or_image)
    else:
        return biascorrection_image(file_or_image)

def biascorrection_image(image):
    MAX_GRAY_VALUE = 600
    if not type(image) == sitk.SimpleITK.Image:
        # normalize values
        image = image*MAX_GRAY_VALUE/image.max()
        image = sitk.GetImageFromArray(image)
        image=sitk.Cast(image, sitk.sitkFloat32)
    else:
        image = sitk.GetArrayFromImage(image)
        image = image * MAX_GRAY_VALUE / image.max()
        image = sitk.GetImageFromArray(image)
        image = sitk.Cast(image, sitk.sitkFloat32)

    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4
    numberOfIteration = [50]
    corrector.SetMaximumNumberOfIterations(numberOfIteration * numberFittingLevels)
    output = corrector.Execute(image, maskImage)
    img2 = sitk.GetArrayFromImage(output)
    return img2

def biascorrection_file(nifti_file):
    inputImage = sitk.ReadImage(nifti_file,sitk.sitkFloat32) 
    return biascorrection_image(inputImage)
