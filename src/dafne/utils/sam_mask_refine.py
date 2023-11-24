# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Dafne-Imaging Team

import numpy as np
from scipy.ndimage import binary_erosion, label
from skimage.morphology import area_opening
import matplotlib.pyplot as plt
import time

from ..segment_anything import sam_model_registry, SamPredictor
import torch
import os
from typing import Callable, Optional
import requests

from ..config import GlobalConfig

CHECKPOINT_SIZES = {
    'vit_h': 2564550879,
    'vit_l': 1249524607,
    'vit_b': 375042383,
}

CHECKPOINT_REMOTE_PATHS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
}

predictor = None
old_img = None


def load_sam(model_type, progress_callback: Optional[Callable[[int, int], None]] = None):
    checkpoint_path = os.path.join(GlobalConfig['MODEL_PATH'], model_type + '.pth')

    try:
        size = os.path.getsize(checkpoint_path)
    except FileNotFoundError:
        size = 0

    if size != CHECKPOINT_SIZES[model_type]:
        print('Downloading SAM checkpoint...')
        # model needs to be downloaded
        r = requests.get(CHECKPOINT_REMOTE_PATHS[model_type], stream=True)
        if r.ok:
            success = True
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            print("Size to download:", total_size_in_bytes)
            block_size = 1024 * 1024  # 1 MB
            current_size = 0
            with open(checkpoint_path, 'wb') as file:
                for data in r.iter_content(block_size):
                    current_size += len(data)
                    if progress_callback is not None:
                        progress_callback(current_size, total_size_in_bytes)
                    file.write(data)

            print("Downloaded size", current_size)
            if current_size != total_size_in_bytes:
                print("Download failed!")
                raise requests.ConnectionError("Error downloading model checkpoint")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    if GlobalConfig['SAM_USE_CUDA'] and torch.cuda.is_available():
        sam.to(device='cuda')

    predictor = SamPredictor(sam)
    return predictor


def dice_score(mask1, mask2):
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)

    # Calculate Dice score
    dice = 2. * intersection.sum() / (mask1_sum + mask2_sum)

    return dice


def enlarge_bounding_box(binary_mask, enlargement_factor=0.2):
    # Find the coordinates of the non-zero elements
    rows, cols = np.where(binary_mask)

    # Determine the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Calculate the enlargement amount
    height = max_row - min_row
    width = max_col - min_col
    expand_height = int(height * enlargement_factor) // 2
    expand_width = int(width * enlargement_factor) // 2

    # Enlarge the bounding box
    min_row = max(0, min_row - expand_height)
    max_row = min(binary_mask.shape[0], max_row + expand_height)
    min_col = max(0, min_col - expand_width)
    max_col = min(binary_mask.shape[1], max_col + expand_width)

    return np.array([min_col, min_row, max_col, max_row])


def generate_points_from_mask(mask):
    npixel = np.sum(mask)
    accumulated_pixels = np.zeros_like(mask)
    begin_time = time.perf_counter()
    while npixel > 0:
        t = time.perf_counter()
        mask = binary_erosion(mask)
        mask_opened = area_opening(mask, 9)
        isolated_pixels = np.logical_and(mask, np.logical_not(mask_opened))
        if np.any(isolated_pixels):
            accumulated_pixels = np.logical_or(accumulated_pixels, isolated_pixels)
        mask = mask_opened
        npixel = np.sum(mask)
        elapsed_time = time.perf_counter() - t
        # plt.imshow(mask)
        # plt.pause(0.1)

    # Label connected components
    labeled_array, num_features = label(accumulated_pixels)

    # Create an output array initialized to zero
    output_map = np.zeros_like(accumulated_pixels)

    point_list = []

    # Iterate through each connected component
    for component in range(1, num_features + 1):
        # Find the coordinates of the voxels in the current component
        voxels = np.argwhere(labeled_array == component)

        # Select the first voxel (or any other voxel) from the component
        if voxels.size > 0:
            point_list.append([voxels[0][1], voxels[0][0]])

    total_time = time.perf_counter() - begin_time
    # print('Total time', total_time)

    return np.array(point_list)


def enhance_mask(img, mask, progress_callback: Optional[Callable[[int, int], None]] = None):
    global predictor, old_img

    # if there is no mask, return the original mask
    if not mask.any():
        return mask

    if GlobalConfig['SAM_MODEL'] == 'Large':
        model_type = 'vit_h'
    elif GlobalConfig['SAM_MODEL'] == 'Medium':
        model_type = 'vit_l'
    elif GlobalConfig['SAM_MODEL'] == 'Small':
        model_type = 'vit_b'

    def show_progress(current, maximum):
        if progress_callback is not None:
            progress_callback(current, maximum)

    show_progress(0, 100)

    if predictor is None:
        print('Loading SAM...')
        predictor = load_sam(model_type, progress_callback)

    show_progress(30, 100)

    if img is not old_img:
        print('Loading image...')
        old_img = img
        img_norm = img * 255 / img.max()
        predictor.set_image(np.stack([img_norm, img_norm, img_norm], 2).astype(np.uint8))

    show_progress(80, 100)

    point_list = generate_points_from_mask(mask)
    print("Positive points", point_list)
    bbox = enlarge_bounding_box(mask, GlobalConfig['SAM_BBOX_EXPAND_FACTOR'])

    # generate negative points
    negative_mask = np.logical_not(mask)
    bbox_region = np.zeros_like(mask)
    bbox_region[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    negative_mask = np.logical_and(negative_mask, bbox_region)

    negative_point_list = generate_points_from_mask(negative_mask)
    print("Negative points", negative_point_list)

    t = time.perf_counter()

    labels = np.array([1] * point_list.shape[0] + [0] * negative_point_list.shape[0])

    if point_list.ndim < 2 and negative_point_list.ndim < 2:
        # there is no point defined. This shouldn't happen, but then return the original mask
        print('Error detecting points')
        return mask
    if point_list.ndim < 2:
        # there is no positive point in the image
        point_list = negative_point_list
    elif negative_point_list.ndim == 2:
        # there are both positive and negative points
        point_list = np.concatenate([point_list, negative_point_list], axis=0)

    # otherwise, we just stay with the positive points. The labels should be fine, because the shape is correct

    masks, scores, logits = predictor.predict(
        point_coords=point_list,
        point_labels=labels,
        box=bbox[None, :],
        multimask_output=True
    )

    elapsed_time = time.perf_counter() - t

    # print('Prediction time', elapsed_time*1000)

    max_dice = 0
    best_mask = 0

    show_progress(100, 100)

    # get the mask closest to the first
    n_output_masks = masks.shape[0]
    for m_id in range(n_output_masks):
        dice = dice_score(masks[m_id, :, :], mask)
        # print('Dice of mask', m_id, dice)
        if dice > max_dice:
            max_dice = dice
            best_mask = m_id

    return masks[best_mask, :, :]