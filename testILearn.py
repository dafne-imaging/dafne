#!/usr/bin/env python3

from dl.LocalModelProvider import LocalModelProvider
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
    slice_range = range(15,25)
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
