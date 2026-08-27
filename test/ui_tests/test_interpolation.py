"""
Mask interpolation: single-slice interpolation and block interpolation between
segmented slices, for the current ROI and for all ROIs, and the toolbox emission
of the selected interpolation method.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, wait_threads, temp_dir,
                    set_active_roi, make_test_volume, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 6)

from dafne.ui.ToolboxWindow import ToolboxWindow

bundle = make_bundle(os.path.join(tmpdir, 'interp.npz'), make_test_volume(SHAPE))
win.loadDirectory(bundle)
process_events()


def square_mask(r0, r1, c0, c1):
    mask = np.zeros(SHAPE[:2], dtype=np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


def test_interpolate_single_slice():
    win.addRoi('interp')
    process_events()
    # segment slices 0 and 2 with squares of different size, then interpolate slice 1
    win.roiManager.set_mask('interp', 0, square_mask(8, 20, 8, 20))
    win.roiManager.set_mask('interp', 2, square_mask(10, 18, 10, 18))
    win.displayImage(1, redraw=False)
    win.redraw()
    process_events()
    win.do_interpolate(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, False)
    process_events()
    interpolated = win.roiManager.get_mask('interp', 1)
    assert np.any(interpolated), 'interpolated slice should not be empty'
    area_0 = win.roiManager.get_mask('interp', 0).sum()
    area_2 = win.roiManager.get_mask('interp', 2).sum()
    assert min(area_0, area_2) <= interpolated.sum() <= max(area_0, area_2), \
        'interpolated area should lie between the two bounding areas'


def test_interpolate_all_rois():
    win.addRoi('interp2')
    process_events()
    win.roiManager.set_mask('interp2', 0, square_mask(2, 8, 2, 8))
    win.roiManager.set_mask('interp2', 2, square_mask(2, 8, 2, 8))
    win.roiManager.set_mask('interp2', 1, np.zeros(SHAPE[:2], dtype=np.uint8))
    win.roiManager.set_mask('interp', 1, np.zeros(SHAPE[:2], dtype=np.uint8))
    win.displayImage(1, redraw=False)
    win.redraw()
    set_active_roi(win, 'interp')
    win.do_interpolate(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, True)
    process_events()
    assert np.any(win.roiManager.get_mask('interp', 1)), 'active ROI should be interpolated'
    assert np.any(win.roiManager.get_mask('interp2', 1)), 'all-ROIs must include the non-active ROI'


def test_interpolate_block():
    win.addRoi('block')
    process_events()
    win.roiManager.set_mask('block', 0, square_mask(8, 20, 8, 20))
    win.roiManager.set_mask('block', 4, square_mask(8, 20, 8, 20))
    # the current slice must lie inside the block (a segmented slice above and one below)
    win.displayImage(2, redraw=False)
    win.redraw()
    set_active_roi(win, 'block')
    win._interpolate_block(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, False, inplace=True)
    process_events()
    for sl in (1, 2, 3):
        assert np.any(win.roiManager.get_mask('block', sl)), \
            'slice {} should be filled by block interpolation'.format(sl)


def test_toolbox_interpolation_method_emission():
    """ The interpolation style combo must map to the right method constants
        (regression test: the SAM entry used to fall through to plain interpolation). """
    combo_mapping = ((0, ToolboxWindow.INTERPOLATE_MASK_SAM),
                     (1, ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE),
                     (2, ToolboxWindow.INTERPOLATE_MASK_REGISTER),
                     (3, ToolboxWindow.INTERPOLATE_MASK_BOTH))
    assert tb.interpolationStyleCombo.count() == len(combo_mapping)
    for index, expected in combo_mapping:
        tb.interpolationStyleCombo.setCurrentIndex(index)
        process_events()
        assert tb._get_interpolation_style() == expected, \
            'combo index {} should map to {}'.format(index, expected)

    # signal emission: detach the main window first, so emitting does not actually
    # run an interpolation in the background
    tb.interpolate_mask.disconnect()
    captured = []
    tb.interpolate_mask.connect(lambda method, all_rois: captured.append((method, all_rois)))
    tb.interpolationStyleCombo.setCurrentIndex(1)
    tb.checkBox_Interpolate_all_ROIs.setChecked(True)
    tb.interpolate_emit()
    assert captured == [(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, True)]
    tb.checkBox_Interpolate_all_ROIs.setChecked(False)
    captured.clear()
    tb.interpolate_emit()
    assert captured == [(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, False)]
    # restore the normal wiring
    tb.interpolate_mask.disconnect()
    tb.interpolate_mask.connect(win.interpolate)


if __name__ == '__main__':
    common.run_tests(globals())
