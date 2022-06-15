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

import numpy as np
import os
import skimage
from skimage.morphology import square
from .padorcut import padorcut
import ..dl.common.biascorrection as biascorrection
from scipy.ndimage import zoom
import math
from skimage.filters import threshold_otsu
import tensorflow as tf
from tensorflow.keras import backend as K


#convert an array of masks into a single numbered mask
def to_mask(categorical_mask,dim,ch):  ##ch: number of labels. ch = 13 for thigh, 7 for leg
    segmentation_mask=np.zeros((dim,dim))
    for i in range(categorical_mask.shape[1]):
        for j in range(categorical_mask.shape[1]):
            segmentation_mask[i,j]=categorical_mask[i,j,1:ch+1].argmax()
    return segmentation_mask


def split_mirror(image):
    '''
    Function to split the training data into left and mirrored right parts cropped and zoomed/padorcutted. 
    '''
    import skimage
    from skimage.morphology import square
    from skimage.filters import threshold_local
    block_size = 15
    # convert image to binary mask
    local_thresh = threshold_local(image, block_size, offset=10)
    binary_local = image > local_thresh
    binary_local=binary_local==0.0
    # remove small black and white spots
    binary_local=skimage.morphology.area_opening(binary_local,area_threshold=4) #20
    binary_local=skimage.morphology.area_closing(binary_local,area_threshold=4) #20
    mountainsc=binary_local[:,:].sum(axis=0)
    mountainsr=binary_local[:,:].sum(axis=1)
    s=0
    p=0
    ii=0 
    jj=0
    a1=0
    a2=0
    a3=0
    a4=0
    b1=0
    b2=0
    #iterate through columns and find transition (black)-white-black-white-black
    while s!=4 and ii<432:
        if mountainsc[ii]>0 and s==0:
            s=1
            a1=ii
        if mountainsc[ii]==0 and s==1 and (ii-a1)>20:
            s=2
            a2=ii
        if mountainsc[ii]>0 and s==2:
            s=3
            a3=ii
        if mountainsc[ii]==0 and s==3 and (ii-a3)>20:
            s=4
            a4=ii
        elif mountainsc[ii]==0 and s==3 and (ii-a3)<20:
            s=2
        ii+=1
        # this means that there is only one patch (legs as attached). Set the splitting point in the middle
        if ii==432 and s==2:
            s=4
            a4=a2
            a2=np.ceil((a4-a1)/2)+a1
            a3=np.ceil((a4-a1)/2)+a1
        if ii==432 and s==3:
            s=4
            a4=431
            #a2=np.ceil((a4-a1)/2)+a1
            #a3=np.ceil((a4-a1)/2)+a1
        if ii==432 and s==1:
            s=4
            a4=431
            a2=np.ceil((a4-a1)/2)+a1
            a3=np.ceil((a4-a1)/2)+a1
    # iterate over columns and find the (black)-white-black transition
    while p!=2:
        if mountainsr[jj]>0 and p==0:
            p=1
            b1=jj
        if mountainsr[jj]==0 and p==1 and (jj-b1)>50:
            p=2
            b2=jj
        jj+=1
    return a1,a2,a3,a4,b1,b2

# returns arrays with number of pixels corresponding to a class and total number of pixels of the images containing that class
def compute_class_frequencies(path,dim,card,ch):
    classes=[0]*ch
    images_containing_class=[0]*ch
    for j in range(1,card+1):
        arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
        if arr.shape[2]==2:
            seg=arr[:,:,1] 
        else:
            seg=to_mask(arr,dim,ch)
        for class_ in range(ch):
            n_pixels=(seg==float(class_)).sum()
            if n_pixels>0:
                classes[class_]+=n_pixels
                images_containing_class[class_]+=dim**2
    return classes,images_containing_class


def calc_weight(img, seg, av, freq, dim, band, ch):
    W = np.zeros((seg.shape[0], seg.shape[1]), dtype='float32')
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh  # image to binary mask
    binary_mask = binary_global == 1.0
    maximum_l = min(3 * np.sqrt(band), seg.shape[0] - 1)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if (binary_mask[i, j] > 0):
                h = 0
                k = 0
                d1 = 0
                d2 = 0
                aa = seg[i, j]
                l = 1
                while h != 1 or k != 1:
                    # find a border with another patch
                    if h == 0:  # point at [i+-l,j] or [i,j+-l] is neither 0 nor aa
                        if (seg[max(i - l, 0), j] != aa and int(seg[max(i - l, 0), j] != 0)):
                            d1 = l
                            h = 1
                            bb = seg[i - l, j]
                        if (seg[min(i + l, seg.shape[0] - 1), j] != aa and int(
                                seg[min(i + l, seg.shape[0] - 1), j]) != 0):
                            d1 = l
                            h = 1
                            bb = seg[min(i + l, seg.shape[0] - 1), j]
                        if (seg[i, max(j - l, 0)] != aa and int(seg[i, max(j - l, 0)]) != 0):
                            d1 = l
                            h = 1
                            bb = seg[i, j - l]
                        if (seg[i, min(j + l, seg.shape[0] - 1)] != aa and int(
                                seg[i, min(j + l, seg.shape[0] - 1)]) != 0):
                            d1 = l
                            h = 1
                            bb = seg[i, min(j + l, seg.shape[0] - 1)]
                    if h == 1 and k == 0 and (
                    # points at [i+-l, j] and [i, j+-l] are neither aa nor bb nor 0. Distance from another patch?
                            (seg[max(i - l, 0), j] != aa and seg[max(i - l, 0), j] != bb and int(
                                seg[max(i - l, 0), j] != 0)) or
                            (seg[min(i + l, seg.shape[0] - 1), j] != aa and seg[
                                min(i + l, seg.shape[0] - 1), j] != bb and int(
                                seg[min(i + l, seg.shape[0] - 1), j]) != 0) or
                            (seg[i, max(j - l, 0)] != aa and seg[i, max(j - l, 0)] != bb and int(
                                seg[i, max(j - l, 0)]) != 0) or
                            (seg[i, min(j + l, seg.shape[0] - 1)] != aa and seg[
                                i, min(j + l, seg.shape[0] - 1)] != bb and int(
                                seg[i, min(j + l, seg.shape[0] - 1)]) != 0)
                    ):
                        d2 = l
                        k = 1
                    if l >= maximum_l:  # l==seg.shape[0]-1:
                        d1 = seg.shape[0]
                        h = 1
                        k = 1
                    l += 1
                W[i, j] = 10 * math.exp(-((d1 + d2) ** 2) / (2 * band))
            W[i, j] = W[i, j] + av / freq[int(seg[i, j])] 
    return W


def input_creation(path,card,dim,band,ch):
    '''
    Creates the training data with labels categorization and creation of weights maps.
    '''
    classes,images=compute_class_frequencies(path,dim,card,ch)
    frequencies=[]
    for cla,ima in zip(classes,images):
        frequencies.append(cla/ima)
    av=sum(frequencies)/ch
    for j in range(1,card+1):
        arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
        img=arr[:,:,0]
        if arr.shape[2]==2:
           seg=arr[:,:,1] 
        else:
           seg=to_mask(arr,dim,ch)
        categ=categorical_and_weight(img,seg,av,frequencies,dim,band,ch)
        arr=np.concatenate([np.reshape(img,(dim,dim,1)),categ],axis=-1)
        np.save(os.path.join(path,'train_'+str(j)+'.npy'),arr)


def calc_aggregated_masks_and_remove_overlap(masks):
    dim = masks.shape[0]
    ch = masks.shape[2]
    # remove overlaps between masks
    cumulated_mask = masks[:, :, 0]
    aggregated_mask = np.zeros_like(cumulated_mask)  # this mask has values from 0 to ch
    for mask_dim in range(1, ch):
        masks[:, :, mask_dim] = np.logical_and(masks[:, :, mask_dim], np.logical_not(cumulated_mask))
        aggregated_mask += mask_dim * masks[:, :, mask_dim]
        cumulated_mask = np.logical_or(cumulated_mask, masks[:, :, mask_dim])

    # set the background mask to the not of the other masks
    masks[:,:,0] = np.logical_not(cumulated_mask)

    return aggregated_mask, masks

def input_creation_mem(image_list: list, mask_list: list, band: float):
    """
    Creates the training data in memory

    Inputs:
        image_list: list of 2D slices (input data)
        mask_list: list of 3D np arrays (stacks of segmented 2D masks). All the masks must have the same number of layers (ROIs)
        band: scalar parameter for the calculation of the weights

    Output:
         list of 3D arrays where: arr[:,:,0] are the base images, arr[:,:,-1] are the weights, and the dimensions in between are the masks
    """

    dim = mask_list[0].shape[0]
    ch = mask_list[0].shape[2]

    aggregated_masks = []
    mask_list_no_overlap = []
    for masks in mask_list:
        agg, new_masks = calc_aggregated_masks_and_remove_overlap(masks)
        aggregated_masks.append(agg)
        mask_list_no_overlap.append(new_masks)

    # compute class frequencies
    classes = [0] * ch
    images_containing_class = [0] * ch
    for seg in aggregated_masks:
        for class_ in range(ch):
            n_pixels = (seg == float(class_)).sum()
            if n_pixels > 0:
                classes[class_] += n_pixels
                images_containing_class[class_] += dim ** 2

    frequencies = []
    for cla, ima in zip(classes, images_containing_class):
        if cla == 0:
            frequencies.append(0)
        else:
            frequencies.append(cla / ima)

    print("Frequencies", frequencies)

    av = sum(frequencies) / ch

    output_data = []

    for slice_number in range(len(image_list)):
        img = image_list[slice_number]
        seg = aggregated_masks[slice_number]
        weight = calc_weight(img,seg,av,frequencies,dim,band,ch)
        arr=np.concatenate([np.reshape(img,(dim,dim,1)),mask_list_no_overlap[slice_number], np.reshape(weight, (dim,dim,1))], axis=-1)
        output_data.append(arr)

    return output_data

def common_input_process(inverse_label_dict, MODEL_RESOLUTION, MODEL_SIZE, trainingData, trainingOutputs):
    # inverse_label_dict = {v: k for k, v in LABELS_DICT.items()}
    nlabels = len(set(inverse_label_dict.values()))+1 # get the number of unique values in the inverse dict
    min_defined_rois = nlabels/2 # do not add to the training set if less than this number of ROIs are defined
    resolution = np.array(trainingData['resolution'])
    zoomFactor = resolution / MODEL_RESOLUTION

    image_list = []
    mask_list = []
    for imageIndex in range(len(trainingData['image_list'])):
        mask_dataset = np.zeros((MODEL_SIZE[0], MODEL_SIZE[1], nlabels))
        defined_rois = 0
        for label, mask in trainingOutputs[imageIndex].items():
            xl=label.endswith('_L')
            xr=label.endswith('_R')
            base_label = label[:-2]
            if xr:
                if (base_label + '_L') in trainingOutputs[imageIndex]:
                    continue # if laterality is split, merge the right into the left
                label = base_label
            if xl:
                if (base_label + '_R') in trainingOutputs[imageIndex]:
                    mask=np.logical_or(trainingOutputs[imageIndex][base_label +'_L'], trainingOutputs[imageIndex][base_label +'_R'])
                label = base_label

            if label not in inverse_label_dict: continue

            if np.sum(mask) > 5:
                defined_rois += 1
            else:
                continue # avoid adding empty masks

            mask = skimage.morphology.area_opening(mask, area_threshold=4)
            mask = skimage.morphology.area_closing(mask, area_threshold=4)
            mask_dataset[:, :, int(inverse_label_dict[label])] = padorcut(zoom(mask, zoomFactor, order=0), MODEL_SIZE)

        if defined_rois > min_defined_rois:
            mask_list.append(mask_dataset)
            image = trainingData['image_list'][imageIndex]
            image = skimage.morphology.area_opening(image, area_threshold=4)
            image = skimage.morphology.area_closing(image, area_threshold=4)
            image_list.append(padorcut(zoom(image, zoomFactor), MODEL_SIZE))

    return image_list, mask_list

def common_input_process_split(inverse_label_dict, MODEL_RESOLUTION, MODEL_SIZE, MODEL_SIZE_SPLIT, trainingData, trainingOutputs):
    nlabels = len(set(inverse_label_dict.values()))+1 # get the number of unique values in the inverse dict
    min_defined_rois = nlabels/2 # do not add to the training set if less than this number of ROIs are defined
    resolution = np.array(trainingData['resolution'])
    zoomFactor = resolution / MODEL_RESOLUTION

    image_list = []
    mask_list = []
    for imageIndex in range(len(trainingData['image_list'])):
        mask_dataset_left = np.zeros((MODEL_SIZE_SPLIT[0], MODEL_SIZE_SPLIT[1], nlabels))
        mask_dataset_right = np.zeros((MODEL_SIZE_SPLIT[0], MODEL_SIZE_SPLIT[1], nlabels))
        defined_rois = 0

        # first, count the defined ROIS, before doing lengthy calculations
        for label, mask in trainingOutputs[imageIndex].items():
            xl = label.endswith('_L')
            xr = label.endswith('_R')
            base_label = label[:-2]
            if xr:
                if (base_label + '_L') in trainingOutputs[imageIndex]:
                    continue  # if laterality is split, merge the right into the left
                label = base_label
            if xl:
                if (base_label + '_R') in trainingOutputs[imageIndex]:
                    mask = np.logical_or(trainingOutputs[imageIndex][base_label + '_L'],
                                         trainingOutputs[imageIndex][base_label + '_R'])
                label = base_label

            if label not in inverse_label_dict: continue

            if np.sum(mask) > 5:
                defined_rois += 1
            else:
                continue # avoid adding empty masks
           
        if defined_rois > min_defined_rois:
            image = trainingData['image_list'][imageIndex]
            image = skimage.morphology.area_opening(image, area_threshold=4)
            image = skimage.morphology.area_closing(image, area_threshold=4)
            image=padorcut(zoom(image, zoomFactor), MODEL_SIZE)
            imgbc= biascorrection.biascorrection_image(image)
            a1,a2,a3,a4,b1,b2=split_mirror(imgbc)
            left=imgbc[int(b1):int(b2),int(a1):int(a2)]
            left=padorcut(left, MODEL_SIZE_SPLIT)
            right=imgbc[int(b1):int(b2),int(a3):int(a4)]
            right=right[::1,::-1]
            right=padorcut(right, MODEL_SIZE_SPLIT)

            for label, mask in trainingOutputs[imageIndex].items():
                xl = label.endswith('_L')
                xr = label.endswith('_R')
                base_label = label[:-2]
                if xr:
                    if (base_label + '_L') in trainingOutputs[imageIndex]:
                        continue  # if laterality is split, merge the right into the left
                    label = base_label
                if xl:
                    if (base_label + '_R') in trainingOutputs[imageIndex]:
                        mask = np.logical_or(trainingOutputs[imageIndex][base_label + '_L'],
                                             trainingOutputs[imageIndex][base_label + '_R'])
                    label = base_label

                if label not in inverse_label_dict: continue

                mask = skimage.morphology.area_opening(mask, area_threshold=4)
                mask = skimage.morphology.area_closing(mask, area_threshold=4)
                mask = padorcut(zoom(mask, zoomFactor, order=0), MODEL_SIZE)
                roileft=mask[int(b1):int(b2),int(a1):int(a2)]
                roileft=padorcut(roileft, MODEL_SIZE_SPLIT)
                roiright=mask[int(b1):int(b2),int(a3):int(a4)]
                roiright=roiright[::1,::-1]
                roiright=padorcut(roiright, MODEL_SIZE_SPLIT)

                mask_dataset_left[:, :, int(inverse_label_dict[label])] = roileft
                mask_dataset_right[:, :, int(inverse_label_dict[label])] = roiright

            image_list.append(left)
            mask_list.append(mask_dataset_left)
            image_list.append(right)
            mask_list.append(mask_dataset_right)

    return image_list, mask_list


def common_input_process_single(inverse_label_dict, MODEL_RESOLUTION, MODEL_SIZE, MODEL_SIZE_SPLIT, trainingData,
                               trainingOutputs, swap):
    nlabels = len(set(inverse_label_dict.values())) + 1  # get the number of unique values in the inverse dict
    min_defined_rois = nlabels / 2  # do not add to the training set if less than this number of ROIs are defined
    resolution = np.array(trainingData['resolution'])
    zoomFactor = resolution / MODEL_RESOLUTION

    image_list = []
    mask_list = []
    for imageIndex in range(len(trainingData['image_list'])):
        mask_dataset = np.zeros((MODEL_SIZE_SPLIT[0], MODEL_SIZE_SPLIT[1], nlabels))
        defined_rois = 0

        # first, count the defined ROIS, before doing lengthy calculations
        for label, mask in trainingOutputs[imageIndex].items():
            xl = label.endswith('_L')
            xr = label.endswith('_R')
            if xl or xr:
                base_label = label[:-2]
            else:
                base_label = label

            if base_label not in inverse_label_dict: continue

            if np.sum(mask) > 5:
                defined_rois += 1
                print(base_label, 'Found, total rois:', defined_rois)
            else:
                continue  # avoid adding empty masks

        if defined_rois > min_defined_rois:
            image = trainingData['image_list'][imageIndex]
            image = skimage.morphology.area_opening(image, area_threshold=4)
            image = skimage.morphology.area_closing(image, area_threshold=4)
            image = padorcut(zoom(image, zoomFactor), MODEL_SIZE_SPLIT)
            imgbc = biascorrection.biascorrection_image(image)

            if swap:
                imgbc = imgbc[::1,::-1]

            for label, mask in trainingOutputs[imageIndex].items():
                xl = label.endswith('_L')
                xr = label.endswith('_R')
                if xl or xr:
                    base_label = label[:-2]
                else:
                    base_label = label

                if base_label not in inverse_label_dict: continue

                mask = skimage.morphology.area_opening(mask, area_threshold=4)
                mask = skimage.morphology.area_closing(mask, area_threshold=4)
                mask = padorcut(zoom(mask, zoomFactor, order=0), MODEL_SIZE_SPLIT)

                if swap:
                    mask = mask[::1,::-1]

                mask_dataset[:, :, int(inverse_label_dict[base_label])] = mask

            image_list.append(imgbc)
            mask_list.append(mask_dataset)

    return image_list, mask_list


def weighted_loss(y_true,y_pred):
    weight_matrix=K.flatten(y_pred[:,:,:,-1])
    y_pre=y_pred[:,:,:,:-1]
    E=-1/(1)*K.dot(K.transpose(K.expand_dims(weight_matrix,axis=-1)),K.expand_dims(K.log(K.flatten(tf.math.reduce_sum(tf.multiply(y_true,y_pre),-1))),axis=-1))
    return E[:,0]
