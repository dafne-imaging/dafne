"""
ROI management and mask editing: add/select/copy/combine/delete ROIs, mask operations
(grow, shrink, despeckle, fill holes, clear), remove overlap, undo/redo, contour mode.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, temp_dir, set_active_roi,
                    make_test_volume, make_square_mask, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 6)

from dafne.ui.ToolboxWindow import ToolboxWindow

bundle = make_bundle(os.path.join(tmpdir, 'edit.npz'), make_test_volume(SHAPE))
win.loadDirectory(bundle)
process_events()
tb.editmode_combo.setCurrentText(ToolboxWindow.EDITMODE_MASK)
process_events()


def current_mask_sum():
    return int(win.getCurrentMask().sum())


def test_add_roi():
    win.addRoi('first')
    process_events()
    assert 'first' in win.roiManager.get_roi_names()
    assert win.getCurrentROIName() == 'first'
    assert current_mask_sum() == 0


def test_set_and_get_mask():
    mask = np.zeros(SHAPE[:2], dtype=np.uint8)
    mask[10:20, 10:20] = 1
    win.setCurrentMask(mask)
    assert current_mask_sum() == 100
    assert np.array_equal(win.getCurrentMask().astype(bool), mask.astype(bool))


def test_mask_grow_shrink():
    original = current_mask_sum()
    win.maskGrow()
    grown = current_mask_sum()
    assert grown > original
    win.maskShrink()
    assert current_mask_sum() == original


def test_undo_redo():
    before = current_mask_sum()
    win.maskGrow()  # @snapshotSaver operation
    after = current_mask_sum()
    assert after > before
    assert win.canUndo()
    win.undo()
    process_events()
    assert current_mask_sum() == before, 'undo should restore the previous mask'
    assert win.canRedo()
    win.redo()
    process_events()
    assert current_mask_sum() == after, 'redo should restore the grown mask'
    win.undo()
    process_events()
    assert current_mask_sum() == before


def test_despeckle_and_fill_holes():
    mask = np.zeros(SHAPE[:2], dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[14, 14] = 0  # hole
    mask[2, 2] = 1  # speckle
    win.setCurrentMask(mask)
    win.maskDespeckle(3)
    assert win.getCurrentMask()[2, 2] == 0, 'despeckle should remove the isolated pixel'
    assert win.getCurrentMask()[12, 12] == 1
    win.maskFillHoles(3)
    assert win.getCurrentMask()[14, 14] == 1, 'fill holes should fill the hole'


def test_copy_and_combine_roi():
    base = np.zeros(SHAPE[:2], dtype=np.uint8)
    base[5:15, 5:15] = 1
    win.setCurrentMask(base)

    win.copyRoi('first', 'copy')
    process_events()
    assert 'copy' in win.roiManager.get_roi_names()
    assert np.array_equal(win.roiManager.get_mask('copy', int(win.curImage)).astype(bool),
                          base.astype(bool))

    # shifted second ROI for the combinations
    other = np.zeros(SHAPE[:2], dtype=np.uint8)
    other[10:20, 10:20] = 1
    win.addRoi('second')
    win.setCurrentMask(other)

    combos = {'Union': np.logical_or, 'Intersection': np.logical_and,
              'Exclusion': np.logical_xor}
    for op_name, op in combos.items():
        dest = 'combo_' + op_name
        win.combineRoi('first', 'second', op_name, dest)
        process_events()
        result = win.roiManager.get_mask(dest, int(win.curImage)).astype(bool)
        assert np.array_equal(result, op(base.astype(bool), other.astype(bool))), op_name

    win.combineRoi('first', 'second', 'Subtraction', 'combo_sub')
    process_events()
    result = win.roiManager.get_mask('combo_sub', int(win.curImage)).astype(bool)
    assert np.array_equal(result, base.astype(bool) & ~other.astype(bool))


def test_remove_overlap():
    set_active_roi(win, 'first')
    win.roiRemoveOverlap()
    process_events()
    first = win.roiManager.get_mask('first', int(win.curImage)).astype(bool)
    second = win.roiManager.get_mask('second', int(win.curImage)).astype(bool)
    assert not np.any(first & second), 'overlap should be removed from the other ROIs'
    assert np.any(second), 'the non-overlapping part must survive'


def test_clear_current_roi():
    set_active_roi(win, 'second')
    assert current_mask_sum() > 0
    win.clearCurrentROI()
    assert current_mask_sum() == 0


def test_remove_roi():
    for name in list(win.roiManager.get_roi_names()):
        if name.startswith('combo_') or name == 'copy':
            win.removeRoi(name)
    process_events()
    names = win.roiManager.get_roi_names()
    assert 'copy' not in names and not any(n.startswith('combo_') for n in names)


def test_contour_mode():
    set_active_roi(win, 'first')
    mask = np.zeros(SHAPE[:2], dtype=np.uint8)
    mask[8:22, 8:22] = 1
    win.setCurrentMask(mask)
    tb.editmode_combo.setCurrentText(ToolboxWindow.EDITMODE_CONTOUR)
    process_events()
    assert win.editMode == ToolboxWindow.EDITMODE_CONTOUR
    roi = win.getCurrentROI()
    assert len(roi.knots) > 0, 'mask should be converted to a contour'
    # back to mask mode: the mask must still be there
    tb.editmode_combo.setCurrentText(ToolboxWindow.EDITMODE_MASK)
    process_events()
    assert win.editMode == ToolboxWindow.EDITMODE_MASK
    reconverted = win.getCurrentMask().astype(bool)
    overlap = np.count_nonzero(reconverted & mask.astype(bool))
    assert overlap / mask.sum() > 0.8, 'mask should survive the contour round trip'


if __name__ == '__main__':
    common.run_tests(globals())
