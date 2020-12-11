#!/usr/bin/env python3

from dl.LocalModelProvider import LocalModelProvider
import numpy as np
import pickle

GENERATE_PICKLE = True

model_provider = LocalModelProvider('models')
segmenter = model_provider.load_model('Thigh')

data_in = np.load('testImages/test_data.npy')
segment_in = np.load('testImages/test_segment.npy',allow_pickle=True)

n_slices = data_in.shape[2]
resolution = np.array([1.037037, 1.037037])

LABELS_DICT = {
        1: 'VL',
        2: 'VM',
        3: 'VI',
        4: 'RF',
        5: 'SAR',
        6: 'GRA',
        7: 'AM',
        8: 'SM',
        9: 'ST',
        10: 'BFL',
        11: 'BFS',
        12: 'AL'
    }

if GENERATE_PICKLE:
    print("Generating data")
    slice_range = range(4,n_slices-5) # range(15,25)
    image_list = []
    seg_list = []

    for slice in slice_range:
        image_list.append(data_in[:,:,slice].squeeze())
        seg_dict = {}
        for k, v in LABELS_DICT.items():
            seg_dict[v] = segment_in[slice][v][:,:] # segment_in[roi_name][:,:,slice]
        seg_list.append(seg_dict)

    pickle.dump(seg_list, open('testImages/test_segment.pickle', 'wb'))
    pickle.dump(image_list, open('testImages/test_data.pickle', 'wb'))
else:
    seg_list = pickle.load(open('testImages/test_segment.pickle', 'rb'))
    image_list = pickle.load(open('testImages/test_data.pickle', 'rb'))

print('Performing incremental learning')
segmenter.incremental_learn({'image_list': image_list, 'resolution': resolution}, seg_list)
