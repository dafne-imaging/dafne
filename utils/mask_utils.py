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

import os
import numpy as np
import dosma


def save_dicom_masks(base_path: str, mask_dict: dict, affine, dicom_headers: list):
    dicom_writer = dosma.DicomWriter(num_workers=4)
    for name, mask in mask_dict.items():
        n = name.strip()
        if n == '': n = '_'
        dicom_path = os.path.join(base_path, n)
        try:
            os.makedirs(dicom_path)
        except OSError:
            pass
        medical_volume = dosma.core.MedicalVolume(mask.astype(np.uint16), affine, dicom_headers)
        dicom_writer.save(medical_volume, dicom_path, fname_fmt='image%04d.dcm')


def save_npy_masks(base_path, mask_dict):
    for name, mask in mask_dict.items():
        npy_name = os.path.join(base_path, name + '.npy')
        np.save(npy_name, mask)


def save_npz_masks(filename, mask_dict):
    np.savez_compressed(filename, **mask_dict)


def save_nifti_masks(base_path, mask_dict, affine):
    nifti_writer = dosma.NiftiWriter()
    for name, mask in mask_dict.items():
        nifti_name = os.path.join(base_path, name + '.nii.gz')
        medical_volume = dosma.core.MedicalVolume(mask.astype(np.uint16), affine)
        nifti_writer.save(medical_volume, nifti_name)