#  Copyright (c) 2022 Dafne-Imaging Team

from bisect import bisect

import skimage
from scipy import ndimage
import numpy as np

from .pySplineInterp import SplineInterpROIClass

MAX_CONTOURS = 100

def find_all_contours(original_mask, erode=True):
    mask = original_mask.copy().astype(bool)
    if erode:
        mask = ndimage.binary_erosion(mask)
    mask = ndimage.binary_dilation(mask)

    if not np.any(mask):
        return []

    raw_contours = skimage.measure.find_contours(mask, level=0.5)

    if len(raw_contours) > MAX_CONTOURS:
        print('Too many contours')
        raw_contours = raw_contours[:MAX_CONTOURS]

    contours = []
    for raw_c in raw_contours:
        # find_contours returns (row, col) float arrays; convert to list of tuples
        contour = [tuple(p) for p in raw_c]
        if len(contour) >= 3:
            contours.append(contour)

    return contours


def calc_contour_distance(contour1, contour2):
    arr1 = np.array(contour1)
    arr2 = np.array(contour2)
    contour2_step = len(arr2) / len(arr1)
    indices = (np.arange(len(arr1)) * contour2_step).astype(int)
    distances = np.linalg.norm(arr1 - arr2[indices], axis=1)
    return float(distances.mean()), distances


def invert_point(p):
    return (p[1], p[0])


def contour_to_spline(contour, precision=1):
    MAX_KNOTS = int(len(contour)/8) # make up to 1 knot every 8 contour points

    if len(contour) < 8:
        return None

    spline_out = SplineInterpROIClass()

    contour_added_indices = [0,
                             int(len(contour) / 4),
                             int(len(contour) / 2),
                             int(len(contour) * 3 / 4)]

    spline_out.addKnots([contour[i] for i in contour_added_indices])

    new_contour = spline_out.getCurve(shift_curve=True)
    if new_contour is None:
        return None

    mean_d, distances = calc_contour_distance(contour, new_contour)

    # add up to MAX_KNOTS
    exhausted_segments = []  # (start, end) contour index pairs too short to subdivide
    for n_knots in range(MAX_KNOTS):
        # Re-apply zeroing for segments already known to be unsplittable
        for s, e in exhausted_segments:
            distances[s:e] = 0

        if not np.any(distances):
            break

        max_d_index = np.argmax(distances)
        # Find insertion point in the added indices list
        insertion_point = bisect(contour_added_indices, max_d_index)

        # try adding a knot at any point between the previous and the next index
        start_handle_point = contour_added_indices[insertion_point - 1]
        if insertion_point == len(contour_added_indices):
            end_handle_point = len(contour) - 1
        else:
            end_handle_point = contour_added_indices[insertion_point]
        if end_handle_point - start_handle_point < 2:
            # segment too short to subdivide; skip it and try other segments
            exhausted_segments.append((start_handle_point, end_handle_point))
            distances[start_handle_point:end_handle_point] = 0
            continue
        spline_out.insertKnot(insertion_point, contour[start_handle_point + 1])
        current_d = 1000
        min_point = start_handle_point + 1
        for test_point in range(start_handle_point + 1, end_handle_point):
            if test_point in contour_added_indices:
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
        # exit if average distance is less than precision
        if current_d < precision:
            break

    # Pruning pass: remove knots that don't contribute beyond the required precision.
    # Iterating backwards keeps indices stable as knots are removed.
    i = len(contour_added_indices) - 1
    while i >= 0 and len(spline_out.knots) > 4:
        spline_out.removeKnot(i)
        test_curve = spline_out.getCurve(shift_curve=True)
        keep_removed = test_curve is not None and calc_contour_distance(contour, test_curve)[0] < precision
        if keep_removed:
            contour_added_indices.pop(i)
        else:
            spline_out.insertKnot(i, contour[contour_added_indices[i]])
        i -= 1

    return spline_out.getSimplifiedSpline()


def mask_to_splines(mask, precision=1):
    """
    Convert a mask to a list of splines
    :param mask: 2D numpy array
    :param precision: precision of the spline (average distance between the spline and the contour of the mask)

    :return: list of SplineInterpROIClass
    """
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
        try:
            spline = contour_to_spline(contour_for_spline, precision=precision)
        except:
            print('Error converting contour to spline')
            spline = None
        if spline is not None:
            splines_out.append(spline)

    return splines_out


def mask_to_trivial_splines(mask, spacing=1):
    """
    Convert a mask to a list of trivial splines. These splines have as many knots as the number of pixels in the contour.
    :param mask: 2D numpy array

    :return: list of SplineInterpROIClass
    """
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
        if spacing != 1 and len(contour_for_spline) > 4 * spacing:
            contour_for_spline = contour_for_spline[::spacing]
        spline_out = SplineInterpROIClass()
        spline_out.addKnots(contour_for_spline, checkProximity=False)
        splines_out.append(spline_out)

    return splines_out


def masks_splines_to_splines_masks(splines):
    def find_closest_other_spline(base_spline, other_spline_list):
        # find the spline with the minimum number of knots
        min_n_knots = min([len(s) for s in other_spline_list] + [len(base_spline)])
        base_spline.reduceKnots(min_n_knots)
        min_spline_index = 0
        test_spline = other_spline_list[0].copy()
        test_spline.reduceKnots(min_n_knots)
        min_d, min_shift = base_spline.calcDistance(test_spline)
        for spline_index, spline in enumerate(other_spline_list[1:]):
            test_spline = spline.copy()
            print(len(test_spline))
            print(min_n_knots)
            test_spline.reduceKnots(min_n_knots)
            print(len(test_spline))
            d, shift = base_spline.calcDistance(test_spline)
            if d < min_d:
                min_d = d
                min_shift = shift
                min_spline_index = spline_index + 1
        min_spline = other_spline_list.pop(min_spline_index)
        min_spline.reduceKnots(min_n_knots)
        min_spline.rotateKnotList(min_shift)
        return min_spline

    outer_spline_list = []
    for spline_index in range(len(splines[0])):
        current_spline_list = [splines[0][spline_index]]
        base_spline = splines[0][spline_index]
        for mask_index, mask_spline_list in enumerate(splines[1:]):
            closest_spline = find_closest_other_spline(base_spline, mask_spline_list)
            current_spline_list.append(closest_spline)
        outer_spline_list.append(current_spline_list)

    return outer_spline_list


def mask_average(mask_list, weight_list=None):
    """
    Average a list of masks by converting them to splines and averaging the knots
    :param mask_list: list of 2D numpy arrays
    :return: 2D numpy array
    """
    # splines is a list of list. The outer list is the list of masks, the inner list is the list of splines for each mask
    splines = []
    for mask in mask_list:
        splines.append(mask_to_trivial_splines(mask, spacing=4))

    # check that every mask has the same number of splines
    n_splines = [len(s) for s in splines]
    if not all([n == n_splines[0] for n in n_splines]):
        raise ValueError('All masks must have the same number of contours')

    outer_spline_list = masks_splines_to_splines_masks(splines)

    if weight_list is None:
        weight_list = [1] * len(mask_list)

    def average_spline(spline_list, weight_list):
        spline_out = SplineInterpROIClass()
        for knot_index in range(len(spline_list[0])):
            knot_list = [spline_list[s].getKnot(knot_index) for s in range(len(spline_list))]
            knot_out = np.average(knot_list, axis=0, weights=weight_list)
            spline_out.addKnot((knot_out[0], knot_out[1]), checkProximity=False)
        return spline_out

    mask_out = np.zeros_like(mask_list[0])
    for spline_list in outer_spline_list:
        average = average_spline(spline_list, weight_list)
        mask_out += average.toMask(mask_out.shape, fast=True)

    mask_out = mask_out > 0
    return mask_out
