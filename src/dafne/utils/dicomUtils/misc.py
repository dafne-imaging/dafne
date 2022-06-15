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
import traceback

import pydicom
import pydicom as dicom
import numpy as np
from PyQt5.QtWidgets import QInputDialog
from muscle_bids.dosma_io import NiftiReader
from muscle_bids.dosma_io.io.dicom_io import to_RAS_affine, DicomReader
from scipy.ndimage import map_coordinates
from muscle_bids import MedicalVolume
import os

from .multiframe import is_enhanced_dicom, is_multi_dicom, convert_to_slices, load_multi_dicom


class ConversionError(Exception):
    pass

def loadDicomFile(fname):
    ds = dicom.read_file(fname)
    # rescale dynamic range to 0-4095
    try:
        pixelData = ds.pixel_array.astype(np.float32)
    except:
        ds.decompress()
        pixelData = ds.pixel_array.astype(np.float32)
    ds.PixelData = ""
    return pixelData, ds

def create_affine(sorted_dicoms):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    original (c) icometrix (https://github.com/icometrix/dicom2nifti) - MIT license

    :param sorted_dicoms: list with sorted dicom datasets (from pydicom)
    """

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
    image_orient1 = np.array(sorted_dicoms[0].ImageOrientationPatient)[0:3]
    image_orient2 = np.array(sorted_dicoms[0].ImageOrientationPatient)[3:6]

    delta_r = float(sorted_dicoms[0].PixelSpacing[0])
    delta_c = float(sorted_dicoms[0].PixelSpacing[1])

    image_pos = np.array(sorted_dicoms[0].ImagePositionPatient)

    last_image_pos = np.array(sorted_dicoms[-1].ImagePositionPatient)

    if len(sorted_dicoms) == 1:
        # Single slice
        step = [0, 0, -1]
    else:
        step = (image_pos - last_image_pos) / (1 - len(sorted_dicoms))

    # check if this is actually a volume and not all slices on the same location
    if np.linalg.norm(step) == 0.0:
        raise ConversionError("NOT_A_VOLUME")

    affine = np.array(
        [[-image_orient1[0] * delta_c, -image_orient2[0] * delta_r, -step[0], -image_pos[0]],
         [-image_orient1[1] * delta_c, -image_orient2[1] * delta_r, -step[1], -image_pos[1]],
         [image_orient1[2] * delta_c, image_orient2[2] * delta_r, step[2], image_pos[2]],
         [0, 0, 0, 1]]
    )
    return affine


def dosma_volume_from_path(path, parent_qobject = None, reorient_data = True):
    medical_volume = None
    affine_valid = False
    basepath = ''
    title = ''

    dicom_ext = ['.dcm', '.ima']
    nii_ext = ['.nii', '.gz']
    npy_ext = ['.npy']
    path = os.path.abspath(path)
    _, ext = os.path.splitext(path)

    basename = os.path.basename(path)

    if ext.lower() in npy_ext:
        data = np.load(path).astype(np.float32)
        medical_volume = MedicalVolume(data, np.eye(4))
        affine_valid = False
        basepath = os.path.dirname(path)
        title = basepath

    elif ext.lower() in nii_ext:
        niiReader = NiftiReader()
        medical_volume = niiReader.load(path)
        if reorient_data:
            desired_orientation, accept = QInputDialog.getItem(parent_qobject,
                                                               "Nifti loader",
                                                               "Select orientation",
                                                               ['Axial', 'Sagittal', 'Coronal'],
                                                               editable=False)
            if not accept: return
            if desired_orientation == 'Axial':
                nifti_orient = ('AP', 'RL', 'IS')
            elif desired_orientation == 'Sagittal':
                nifti_orient = ('SI', 'PA', 'RL')
            else:
                nifti_orient = ('SI', 'RL', 'AP')

            medical_volume.reformat(nifti_orient, inplace=True)

        affine_valid = True
        title = os.path.basename(path)
        basepath = os.path.dirname(path)

    else:  # assume it's dicom
        load_dicom_dir = False
        if os.path.isfile(path):
            basepath = os.path.dirname(path)
            dataset = pydicom.read_file(path, force=True)
            if is_enhanced_dicom(dataset):
                if is_multi_dicom(dataset):
                    multi_dicom_dataset = load_multi_dicom(dataset)
                    # this is a multi dicom dataset
                    # let the user choose which dataset to load
                    dataset_key, accept = QInputDialog.getItem(parent_qobject,
                                                               "Multi dicom",
                                                               "Choose dataset to load",
                                                               list(multi_dicom_dataset.keys()),
                                                               editable=False)
                    if not accept: return
                    header_list = multi_dicom_dataset[dataset_key][1]
                    data = multi_dicom_dataset[dataset_key][0].astype(np.float32)
                else:
                    # enhanced dicom but not with multiple contrasts
                    data, header_list = convert_to_slices(dataset)

                affine = to_RAS_affine(header_list)
                medical_volume = MedicalVolume(data, affine, header_list)
                affine_valid = True
                title = os.path.basename(path)
                load_dicom_dir = False
            else:
                title = os.path.basename(basepath)
                basename = ''
                load_dicom_dir = True

        elif os.path.isdir(path):
            basename = ''
            basepath = path
            title = basepath
            load_dicom_dir = True

        if load_dicom_dir:
            try:
                dr = DicomReader(num_workers=0, group_by=None, ignore_ext=True)
                medical_volume = dr.load(basepath)[0]
                affine_valid = True
            except:
                # Error reading with DOSMA. use standard dicom
                print('Error using DOSMA for load')
                #traceback.print_exc()
                l = os.listdir(basepath)
                threeDlist = []
                header_list = []
                for f in sorted(l):
                    if os.path.basename(f).startswith('.'): continue
                    try:
                        dFile = dicom.read_file(os.path.join(basepath, f), force=True)
                    except:
                        print("Error loading", f)
                        continue
                    dFile.ensure_file_meta()
                    if 'TransferSyntaxUID' not in dFile.file_meta:
                        dFile.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

                    try:
                        try:
                            dFile.decompress()
                        except ValueError:
                            dFile.NumberOfFrames = 1
                            dFile.decompress()
                    except:
                        print('Error decompressing file')


                    try:
                        pixelIma = dFile.pixel_array
                    except:
                        print('Error loading pixel array for file', f)
                        continue

                    threeDlist.append(np.copy(pixelIma.astype('float32')))
                    dFile.PixelData = ""
                    header_list.append(dFile)
                data = np.stack(threeDlist, axis=2)
                affine = to_RAS_affine(header_list)
                medical_volume = MedicalVolume(data, affine, header_list)
                affine_valid = True

    return medical_volume, affine_valid, title, basepath, basename


def realign_medical_volume(source: MedicalVolume, destination: MedicalVolume, interpolation_order: int = 3):
    """Realign this volume to the image space of another volume. Similar to ``reformat_as``,
    except that it supports fine rotations, translations and shape changes, so that the affine
    matrix and extent of the modified volume is identical to the one of the target.


    Args:
        source (MedicalVolume): The volume to realign
        destination (MedicalVolume): The realigned volume will have the same extent and affine matrix of ``destination``.
        interpolation_order (int, optional): spline interpolation order.

    Returns:
        MedicalVolume: The realigned volume.
    """

    target = destination.volume
    print("Alignment target shape:", target.shape)

    # disregard our own slice thickness for the moment
    # calculate the difference from the center of each 3D "partition" and the real center of the 2D slice
    z_offset = 0

    # The following would be need to be used if the origin of the affine matrix was calculated on the middle of the
    # slice with thickness == slice spacing. But centering the converted dataset on the real center of the slice
    # seems to be the norm, hence the z_offset is 0
    # z_offset = (other_thickness/other.pixel_spacing[2]/2 - 1/2) #(other_thickness - other.pixel_spacing[2])/2

    shape_as_range = (np.arange(target.shape[0]),
                      np.arange(target.shape[1]),
                      np.arange(target.shape[2]) + z_offset)

    coords = np.array(np.meshgrid(*shape_as_range, indexing='ij')).astype(float)

    # Add additional dimension with 1 to each coordinate to make work with 4x4 affine matrix
    addon = np.ones([1, coords.shape[1], coords.shape[2], coords.shape[3]])
    coords = np.concatenate([coords, addon], axis=0)  # shape: [4, x, y, z]
    coords = coords.reshape([4, -1])  # shape: [4, x*y*z]

    # build affine which maps from target grid to source grid
    aff_transf = np.linalg.inv(source.affine) @ destination.affine

    # transform the coords from target grid to the space of source image
    coords_src = aff_transf @ coords  # shape: [4, x*y*z]

    # reshape to original spatial dimensions
    coords_src = coords_src.reshape((4,) + target.shape)[:3, ...]  # shape: [3, x, y, z]

    # Will create a image with the spatial size of coords_src (with is target.shape). Each
    # coordinate contains a place in the source image from which the intensity is taken
    # and filled into the new image. If the coordinate is not within the range of the source image then
    # will be filled with 0.
    src_transf_data = map_coordinates(source.volume, coords_src, order=interpolation_order)

    mv = MedicalVolume(src_transf_data, destination.affine, destination.headers())

    return mv

