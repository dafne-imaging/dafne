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

import pydicom
import numpy as np


def flatten_data(d, new_dataset=None):
    if new_dataset is None:
        new_dataset = pydicom.Dataset()
        new_dataset.is_little_endian = d.is_little_endian
        new_dataset.is_implicit_VR = d.is_implicit_VR
    for element in d.iterall():
        if not isinstance(element.value, pydicom.sequence.Sequence):
            new_dataset.add(element)
        # else:
        #    print('Skipping', element)
    return new_dataset


def convert_to_slices(data_in):
    d = data_in.copy()
    slice_data = d.pop((0x5200, 0x9230)).value
    try:
        pixel_data = d.pixel_array.transpose([1, 2, 0]).astype(np.float32)
    except:
        d.decompress()
        pixel_data = d.pixel_array.transpose([1, 2, 0]).astype(np.float32)
    d.pop((0x7fe0, 0x0010))
    header_list = []
    for slice_header in slice_data:
        new_slice_header = flatten_data(d)
        flatten_data(slice_header, new_slice_header)
        header_list.append(new_slice_header)
    return pixel_data, header_list


def divide_slice_types(pixel_data, header_list):
    def get_frame_type_philips(h):
        return h[(0x2005, 0x1011)].value

    def get_frame_type_generic(h):
        return ';'.join(h.FrameType)

    get_frame_type = get_frame_type_philips
    try:
        current_frame_type = get_frame_type(header_list[0])
    except KeyError:
        get_frame_type = get_frame_type_generic
        current_frame_type = get_frame_type(header_list[0])

    frames_out = {}
    current_frame_number = 0
    last_frame_number = 0
    for dicom_info in header_list:
        new_frame_type = get_frame_type(dicom_info)
        if new_frame_type != current_frame_type:
            new_header = header_list[last_frame_number:current_frame_number]
            new_data = pixel_data[:, :, last_frame_number:current_frame_number]
            frames_out[current_frame_type] = (new_data, new_header)
            current_frame_type = new_frame_type
            last_frame_number = current_frame_number
        current_frame_number += 1
    return frames_out


def load_multi_dicom(dicom_file):
    d = pydicom.dcmread(dicom_file)
    try:
        number_of_frames = int(d.NumberOfFrames)
    except:
        number_of_frames = 1

    if number_of_frames <= 1:
        return None  # not a multi dicom

    pixel_data, header_list = convert_to_slices(d)
    return divide_slice_types(pixel_data, header_list)