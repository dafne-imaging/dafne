#!/usr/bin/env python3

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

from src.dafne_dl import LocalModelProvider
import numpy as np
import pickle

GENERATE_PICKLE = False

model_provider = LocalModelProvider('models')
segmenter = model_provider.load_model('Thigh')

data_in = np.load('testImages/test_data.npy')
segment_in = np.load('testImages/test_segment.npz')

n_slices = data_in.shape[2]
resolution = np.array([1.037037, 1.037037])

if GENERATE_PICKLE:
    print("Generating data")
    slice_range = range(4,n_slices-5)
    image_list = []
    seg_list = []

    for slice in slice_range:
        image_list.append(data_in[:,:,slice].squeeze())
        seg_dict = {}
        for roi_name in segment_in:
            seg_dict[roi_name] = segment_in[roi_name][:,:,slice]
        seg_list.append(seg_dict)

    pickle.dump(seg_list, open('testImages/test_segment.pickle', 'wb'))
    pickle.dump(image_list, open('testImages/test_data.pickle', 'wb'))
else:
    seg_list = pickle.load(open('testImages/test_segment.pickle', 'rb'))
    image_list = pickle.load(open('testImages/test_data.pickle', 'rb'))

print('Performing incremental learning')
segmenter.incremental_learn({'image_list': image_list, 'resolution': resolution}, seg_list)
