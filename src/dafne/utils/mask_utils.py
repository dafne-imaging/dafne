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
from muscle_bids.dosma_io import DicomWriter, NiftiWriter
from muscle_bids import MedicalVolume
from matplotlib import pyplot as plt


def save_dicom_masks(base_path: str, mask_dict: dict, affine, dicom_headers: list):
    dicom_writer = DicomWriter(num_workers=0)
    for name, mask in mask_dict.items():
        n = name.strip()
        if n == '': n = '_'
        dicom_path = os.path.join(base_path, n)
        try:
            os.makedirs(dicom_path)
        except OSError:
            pass
        medical_volume = MedicalVolume(mask.astype(np.uint16), affine, dicom_headers)
        dicom_writer.save(medical_volume, dicom_path, fname_fmt='image%04d.dcm')


def save_npy_masks(base_path, mask_dict):
    for name, mask in mask_dict.items():
        npy_name = os.path.join(base_path, name + '.npy')
        np.save(npy_name, mask)


def save_npz_masks(filename, mask_dict):
    np.savez_compressed(filename, **mask_dict)


def save_nifti_masks(base_path, mask_dict, affine):
    nifti_writer = NiftiWriter()
    for name, mask in mask_dict.items():
        nifti_name = os.path.join(base_path, name + '.nii.gz')
        medical_volume = MedicalVolume(mask.astype(np.uint16), affine)
        nifti_writer.save(medical_volume, nifti_name)


def make_accumulated_mask(mask_dict):
    accumulated_mask = None
    current_value = 1
    name_list = []
    for name, mask in mask_dict.items():
        name_list.append(name)
        if accumulated_mask is None:
            accumulated_mask = 1*(mask>0)
        else:
            accumulated_mask += current_value * (mask>0)
        current_value += 1
    return accumulated_mask, name_list


def write_legend(filename, name_list):
    with open(filename, 'w') as f:
        f.write('Value,Label\n')
        for index, name in enumerate(name_list):
            f.write(f'{index+1},{name}\n')


def write_itksnap_legend(filename, name_list):
    PREAMBLE = """################################################
# ITK-SnAP Label Description File
# File format: 
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields: 
#    IDX:   Zero-based index 
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    MSH:   Label mesh visibility (0 or 1)
#  LABEL:   Label description 
################################################\n"""

    line_format = '{id:>5d}{red:>6d}{green:>5d}{blue:>5d}{alpha:>9.2g}{vis:>3d}{mesh:>3d}    "{label}"\n'

    cmap = plt.get_cmap('hsv')
    nLabels = len(name_list)
    nColors = cmap.N

    step = int(nColors/nLabels)

    with open(filename, 'w') as f:
        f.write(PREAMBLE)
        f.write(line_format.format(id=0, red=0, green=0, blue=0, alpha=0, vis=0, mesh=0, label='Clear Label'))
        for index, name in enumerate(name_list):
            color = cmap(index*step)
            f.write(line_format.format(id=index+1, red=int(color[0]*255), green=int(color[1]*255), blue=int(color[2]*255),
                    alpha=1, vis=1, mesh=1, label=name))


def save_single_nifti(filename, mask_dict, affine):
    nifti_writer = NiftiWriter()
    accumulated_mask, name_list = make_accumulated_mask(mask_dict)
    legend_name = filename + '.csv'
    snap_legend_name = filename + '_itk-snap.txt'
    medical_volume = MedicalVolume(accumulated_mask.astype(np.uint16), affine)
    nifti_writer.save(medical_volume, filename)
    write_legend(legend_name, name_list)
    write_itksnap_legend(snap_legend_name, name_list)


def save_single_dicom_dataset(base_path, mask_dict, affine, dicom_headers: list):
    dicom_writer = DicomWriter(num_workers=0)
    accumulated_mask, name_list = make_accumulated_mask(mask_dict)
    medical_volume = MedicalVolume(accumulated_mask.astype(np.uint16), affine, dicom_headers)
    try:
        os.makedirs(base_path)
    except OSError:
        pass
    dicom_writer.save(medical_volume, base_path, fname_fmt='image%04d.dcm')
    legend_name = os.path.join(base_path, 'legend.csv')
    snap_legend_name = os.path.join(base_path, 'legend_itk-snap.txt')
    write_legend(legend_name, name_list)
    write_itksnap_legend(snap_legend_name, name_list)