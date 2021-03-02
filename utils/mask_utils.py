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

from .dicomUtils.dicom3D import save3dDicom
import os
import numpy as np
import nibabel as nib


def save_dicom_masks(base_path: str, mask_dict: dict, dicom_headers: list):
    for name, mask in mask_dict.items():
        dicom_path = os.path.join(base_path, name)
        try:
            os.makedirs(dicom_path)
        except OSError:
            pass
        save3dDicom(mask, dicom_headers, dicom_path)


def save_npy_masks(base_path, mask_dict):
    for name, mask in mask_dict.items():
        npy_name = os.path.join(base_path, name + '.npy')
        np.save(npy_name, mask)


def save_npz_masks(filename, mask_dict):
    np.savez_compressed(filename, **mask_dict)


def save_nifti_masks(base_path, mask_dict, affine, transpose=None):
    for name, mask in mask_dict.items():
        nifti_name = os.path.join(base_path, name + '.nii.gz')
        if transpose is not None:
            signs = np.sign(transpose)
            transpose = np.abs(transpose) - 1
            for ax in range(3):
                if signs[ax] < 0:
                    mask = np.flip(mask, axis=ax)
            mask = np.transpose(mask, np.argsort(transpose)) #invert the original transposition
        niimg = nib.Nifti2Image(mask, affine)
        niimg.to_filename(nifti_name)