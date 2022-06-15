#!/usr/bin/env python
# coding: utf-8

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

import numpy as np
import pickle
from generate_thigh_model import coscia_unet as unet
import src.dafne_dl.common.preprocess_train as pretrain

model=unet()
#model.load_weights('weights/weights_coscia.hdf5') ## old
model.load_weights('Weights_incremental/thigh/weights -  5 -  86965.35.hdf5') ## incremental

seg_list = pickle.load(open('testImages/test_segment.pickle', 'rb'))
image_list = pickle.load(open('testImages/test_data.pickle', 'rb'))

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
'''
LABELS_DICT = {
        1: 'SOL',
        2: 'GM',
        3: 'GL',
        4: 'TA',
        5: 'ELD',
        6: 'PE',
        }
'''
MODEL_RESOLUTION = np.array([1.037037, 1.037037])
MODEL_SIZE = (432, 432)

image_list, mask_list = pretrain.common_input_process(LABELS_DICT, MODEL_RESOLUTION, MODEL_SIZE, {'image_list': image_list, 'resolution': MODEL_RESOLUTION}, seg_list)

ch = mask_list[0].shape[2]
aggregated_masks = []
mask_list_no_overlap = []
for masks in mask_list:
    agg, new_masks = pretrain.calc_aggregated_masks_and_remove_overlap(masks)
    aggregated_masks.append(agg)
    mask_list_no_overlap.append(new_masks)

for slice_number in range(len(image_list)):
    img = image_list[slice_number]
    segmentation = model.predict(np.expand_dims(np.stack([img,np.zeros(MODEL_SIZE)],axis=-1),axis=0))
    segmentationnum = np.argmax(np.squeeze(segmentation[0,:,:,:ch]), axis=2)
    cateseg=np.zeros((432,432,ch),dtype='float32')
    for i in range(432):
        for j in range(432):
            cateseg[i,j,int(segmentationnum[i,j])]=1.0
    acc=0
    y_pred=cateseg
    y_true=mask_list_no_overlap[slice_number]
    for j in range(ch):  ## Dice
        elements_per_class=y_true[:,:,j].sum()
        predicted_per_class=y_pred[:,:,j].sum()
        intersection=(np.multiply(y_pred[:,:,j],y_true[:,:,j])).sum()
        intersection=2.0*intersection
        union=elements_per_class+predicted_per_class
        acc+=intersection/(union+0.000001)
    acc=acc/ch
    print(str(slice_number)+'__'+str(acc))
