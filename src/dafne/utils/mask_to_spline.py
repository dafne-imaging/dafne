#  Copyright (c) 2022 Dafne-Imaging Team

from bisect import bisect

import skimage
from scipy import ndimage
import numpy as np

from .pySplineInterp import SplineInterpROIClass

MAX_POINTS = 1000
MAX_CONTOURS = 100


def get_point_context(img, point):
    x, y = point
    context_coords = []
    context_coords.append((x-1, y-1))
    context_coords.append((x-1, y))
    context_coords.append((x-1, y+1))
    context_coords.append((x, y+1))
    context_coords.append((x+1, y+1))
    context_coords.append((x+1, y))
    context_coords.append((x+1, y-1))
    context_coords.append((x, y-1))
    context = [img[c] for c in context_coords]
    #img[point] = 0 # unsure about this
    return context_coords, context


def get_next_coord(context_coords, context):
    bg_found = not context[0]

    def examine_context():
        nonlocal bg_found
        for i, val in enumerate(context):
            if bg_found and val:
                return context_coords[i]
            elif not val:
                bg_found = True
        return None

    next_coord = examine_context()
    if next_coord is None:
        # if we are here, we have not found a next coord.
        # it means we have done a whole circle around the initial point.
        # we must have found a bg point by now, if not, we have a problem
        if not bg_found:
            raise ValueError('Initial point was inside the shape')

        # we have found a bg point, but we have not found a fg point.
        # rerun the procedure, but now the bg point is already found
        next_coord = examine_context()
        if next_coord is None:
            raise ValueError('Initial point was outside the shape')

    return next_coord


def find_contour(mask):
    first_point = np.unravel_index(np.argmax(mask > 0), mask.shape)
    contour_list = [first_point]
    context_coords, context = get_point_context(mask, first_point)
    next_point = get_next_coord(context_coords, context)
    mask[next_point] = True # reset the first point so we can find it again
    while next_point != first_point:
        contour_list.append(next_point)
        if len(contour_list) > MAX_POINTS:
            raise ValueError('Too many points')
        #print(len(contour_list))
        context_coords, context = get_point_context(mask, next_point)
        #print(context)
        #print(context_coords)
        next_point = get_next_coord(context_coords, context)

    return contour_list

def find_all_contours(original_mask, erode=True):
    contours = []
    mask = original_mask.copy()
    if erode:
        mask = ndimage.binary_erosion(mask)
    mask = ndimage.binary_dilation(mask)
    n_contours = 0
    while np.any(mask):
        try:
            contour_points = find_contour(mask)
        except ValueError as e:
            print('Error finding contour')
            break
        contours.append(contour_points)
        contour_image = np.zeros_like(mask)
        for p in contour_points:
            contour_image[p] = 1
        flood_mask = skimage.segmentation.flood(contour_image, (0,0), connectivity=1)
        mask = np.logical_xor(mask, np.logical_not(flood_mask))
        n_contours += 1
        if n_contours > MAX_CONTOURS:
            print('Too many contours')
            break

    return contours


def calc_contour_distance(contour1, contour2):
    distances = []
    contour2_step = float(len(contour2)) / len(contour1)
    for i, p in enumerate(contour1):
        contour2_index = int(i * contour2_step)
        #print(len(contour2), contour2_index, p, contour2[(contour2_index) % len(contour2)])
        distances.append(np.linalg.norm(np.array(p) - np.array(contour2[contour2_index])))

    return np.mean(distances), distances


def invert_point(p):
    return (p[1], p[0])


def contour_to_spline(contour, precision=1):
    MAX_KNOTS = 20

    spline_out = SplineInterpROIClass()

    contour_added_indices = [0,
                             int(len(contour) / 4),
                             int(len(contour) / 2),
                             int(len(contour) * 3 / 4)]

    spline_out.addKnots([contour[i] for i in contour_added_indices])

    mean_d, distances = calc_contour_distance(contour, spline_out.getCurve(shift_curve=True))

    # add up to MAX_KNOTS
    for n_knots in range(MAX_KNOTS):
        # print(distances)
        max_d_index = np.argmax(distances)
        # Find insertion point in the added indices list
        insertion_point = bisect(contour_added_indices, max_d_index)

        # try adding a knot at any point between the previous and the next index
        start_handle_point = contour_added_indices[insertion_point - 1]
        if insertion_point == len(contour_added_indices):
            end_handle_point = len(contour)-1
        else:
            end_handle_point = contour_added_indices[insertion_point]
        if end_handle_point - start_handle_point < 2:
            # we cannot add a knot here
            break
        spline_out.insertKnot(insertion_point, contour[start_handle_point+1])
        current_d = 1000
        min_point = start_handle_point + 1
        for test_point in range(start_handle_point + 1, end_handle_point):
            if test_point in contour_added_indices:
                print('test point already in contour_added_indices')
                continue
            spline_out.replaceKnot(insertion_point, contour[test_point])

            # calculate the new average distance of the whole path
            new_d, new_distances = calc_contour_distance(contour, spline_out.getCurve(shift_curve=True))
            # if the new distance is smaller than the one found, save the new distance and the new point
            if new_d < current_d:
                current_d = new_d
                min_point = test_point
                distances = new_distances

        contour_added_indices.insert(insertion_point, min_point)
        spline_out.replaceKnot(insertion_point, contour[min_point])
        # exit if average distance is less than 1px
        if current_d < precision:
            break
    return spline_out


def mask_to_splines(mask, precision=1):
    contours = find_all_contours(mask)
    # if no contours are found, try if there are only some thin structures
    if not contours:
        contours = find_all_contours(mask, erode=False)
    splines_out = []

    for contour_points in contours:
        # we need to invert because the spline wants coordinates in the "graphics" format, which is
        # transposed with respect to the numpy format
        contour_for_spline = [invert_point(p) for p in contour_points]
        contour_for_spline.reverse()
        splines_out.append(contour_to_spline(contour_for_spline, precision=precision))

    return splines_out