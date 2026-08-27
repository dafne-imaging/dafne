"""
Toolbox window state and controls: edit modes, brush handling, ROI list/combo
synchronization, undo/redo enabling, export enabling, timepoint slider, contrast
selector, splash screen and general enable/disable.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, temp_dir,
                    make_test_volume, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

from dafne.ui.ToolboxWindow import ToolboxWindow

bundle = make_bundle(os.path.join(tmpdir, 'tb.npz'), make_test_volume((32, 32, 4)))
win.loadDirectory(bundle)
process_events()


def test_edit_mode_switch():
    tb.editmode_combo.setCurrentText(ToolboxWindow.EDITMODE_CONTOUR)
    process_events()
    assert tb.get_edit_mode() == ToolboxWindow.EDITMODE_CONTOUR
    assert win.editMode == ToolboxWindow.EDITMODE_CONTOUR
    tb.editmode_combo.setCurrentText(ToolboxWindow.EDITMODE_MASK)
    process_events()
    assert tb.get_edit_mode() == ToolboxWindow.EDITMODE_MASK
    assert win.editMode == ToolboxWindow.EDITMODE_MASK


def test_brush_controls():
    tb.brushsize_slider.setValue(5)
    process_events()
    brush_type, brush_size = tb.get_brush()
    assert brush_size == 5
    assert brush_type in (ToolboxWindow.BRUSH_CIRCLE, ToolboxWindow.BRUSH_SQUARE)
    tb.increase_brush_size()
    process_events()
    assert tb.get_brush()[1] == brush_size + 1
    tb.reduce_brush_size()
    process_events()
    assert tb.get_brush()[1] == brush_size


def test_roi_list_sync():
    win.addRoi('roi_a')
    win.addRoi('roi_b')
    process_events()
    combo_items = [tb.roi_combo.itemText(i) for i in range(tb.roi_combo.count())]
    assert 'roi_a' in combo_items and 'roi_b' in combo_items
    tb.set_current_roi('roi_a', 0)
    assert tb.get_current_roi_subroi()[0] == 'roi_a'
    tb.set_current_roi('roi_b', 0)
    assert tb.get_current_roi_subroi()[0] == 'roi_b'


def test_undo_redo_enabling():
    tb.undo_enable(True)
    assert tb.undoButton.isEnabled()
    tb.undo_enable(False)
    assert not tb.undoButton.isEnabled()
    tb.redo_enable(True)
    assert tb.redoButton.isEnabled()
    tb.redo_enable(False)
    assert not tb.redoButton.isEnabled()


def test_exports_enabled():
    tb.set_exports_enabled(numpy=False, dicom=False, nifti=False)
    assert not tb.menuSave_as_Numpy.isEnabled()
    assert not tb.actionSave_as_Dicom.isEnabled()
    assert not tb.actionSave_as_Nifti.isEnabled()
    tb.set_exports_enabled(numpy=True, dicom=True, nifti=True)
    assert tb.menuSave_as_Numpy.isEnabled()
    assert tb.actionSave_as_Dicom.isEnabled()
    assert tb.actionSave_as_Nifti.isEnabled()


def test_timepoint_slider():
    tb.set_timepoints(5)
    process_events()
    assert tb.timeFrameSlider.maximum() == 4
    assert tb.timeFrameFrame.isVisibleTo(tb), 'time widget should be shown for 4D data'
    captured = []
    tb.timepoint_changed.connect(lambda t: captured.append(t))
    tb.set_current_timepoint(3)
    process_events()
    assert tb.get_current_timepoint() == 3
    assert 3 in captured
    tb.set_timepoints(1)
    process_events()
    assert not tb.timeFrameFrame.isVisibleTo(tb), 'time widget should be hidden for 3D data'


def test_contrast_combo():
    tb.clear_contrast_combo()
    process_events()
    assert tb.find_contrast_in_combo(ToolboxWindow.BASE_CONTRAST_LABEL) >= 0, \
        'base contrast is always present'
    assert not tb.contrastFrame.isVisibleTo(tb), 'selector hidden with a single contrast'
    tb.add_contrast_to_combo('t2')
    process_events()
    assert tb.find_contrast_in_combo('t2') >= 0
    assert tb.contrastFrame.isVisibleTo(tb), 'selector shown with more than one contrast'
    captured = []
    tb.contrast_changed.connect(lambda name: captured.append(name))
    tb.contrastCombo.setCurrentIndex(tb.find_contrast_in_combo('t2'))
    process_events()
    assert 't2' in captured
    tb.remove_contrast_combo('t2')
    process_events()
    assert tb.find_contrast_in_combo('t2') < 0
    assert not tb.contrastFrame.isVisibleTo(tb)


def test_splash():
    tb.set_splash(True, 1, 4, 'Testing')
    process_events()
    assert tb.splashWidget.isVisibleTo(tb)
    tb.set_splash(False, 0, 1)
    process_events()
    assert not tb.splashWidget.isVisibleTo(tb)


def test_general_enable():
    tb.general_enable(False)
    process_events()
    assert not tb.roi_combo.isEnabled() or not tb.mainUIWidget.isEnabled()
    tb.general_enable(True)
    process_events()


if __name__ == '__main__':
    common.run_tests(globals())
