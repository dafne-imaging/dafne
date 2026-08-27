"""
Triplanar/3D viewer integrated with the main interface: population on show,
bidirectional slice synchronization, ROI switching, contrast and time frame updates.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, wait_threads, temp_dir,
                    set_active_roi, make_test_volume, make_square_mask, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

from dafne.ui.Viewer3D import MAIN_PLANE_AXIS
from dafne.ui.ToolboxWindow import ToolboxWindow

SHAPE = (32, 32, 6)
volume = make_test_volume(SHAPE)
extra = np.full(SHAPE, 500.0, dtype=np.float32)
mask_a = make_square_mask(SHAPE, 5, 15, 5, 15)
mask_b = make_square_mask(SHAPE, 18, 28, 18, 28)

bundle = make_bundle(os.path.join(tmpdir, 'base.npz'), volume,
                     {'alpha': mask_a, 'beta': mask_b})
extra_file = make_bundle(os.path.join(tmpdir, 'extra.npz'), extra)

win.loadDirectory(bundle)
win._load_additional_contrast_data(extra_file, 'extra')
wait_threads()

viewer = tb.viewer3D


def main_view():
    return [v for v in viewer.views if v.fixed_axis == MAIN_PLANE_AXIS][0]


def test_viewer_populated_on_show():
    tb.action3D_Viewer.setChecked(True)
    tb.toggle_3D_viewer(True)
    process_events()
    assert viewer.isVisible()
    assert viewer.anatomy is not None and viewer.anatomy.shape == SHAPE
    assert np.array_equal(viewer.data.astype(bool), mask_a.astype(bool)), \
        'active ROI mask should be sent to the viewer'
    assert viewer.other_mask is not None and np.all(viewer.other_mask[mask_b > 0] == 2), \
        'the non-active ROI should be labeled'
    assert viewer.position[MAIN_PLANE_AXIS] == int(win.curImage)


def test_main_to_viewer_slice_sync():
    win.displayImage(3, redraw=False)
    win.redraw()
    process_events()
    assert viewer.position[MAIN_PLANE_AXIS] == 3
    assert int(main_view().curImage) == 3


def test_viewer_to_main_slice_sync():
    viewer.set_position({MAIN_PLANE_AXIS: 1})
    process_events()
    assert int(win.curImage) == 1


def test_roi_change_updates_viewer():
    set_active_roi(win, 'beta')
    wait_threads()
    assert np.array_equal(viewer.data.astype(bool), mask_b.astype(bool))
    assert np.all(viewer.other_mask[mask_a > 0] == 2)


def test_contrast_switch_updates_viewer():
    win.change_contrast('extra')
    process_events()
    assert np.allclose(viewer.anatomy, extra), 'viewer should show the extra contrast'
    win.change_contrast(ToolboxWindow.BASE_CONTRAST_LABEL)
    process_events()
    assert np.allclose(viewer.anatomy, win.medical_volume.volume)


def test_timepoint_switch_updates_viewer():
    nt = 3
    volume_4d = make_test_volume(SHAPE + (nt,), seed=3)
    mask_4d = np.zeros(SHAPE + (nt,), dtype=np.uint8)
    mask_4d[5:15, 5:15, :, :] = 1
    bundle_4d = make_bundle(os.path.join(tmpdir, 'time.npz'), volume_4d, {'m': mask_4d})
    win.loadDirectory(bundle_4d)
    process_events()
    assert np.allclose(viewer.anatomy, volume_4d[..., 0])
    tb.set_current_timepoint(2)
    process_events()
    assert np.allclose(viewer.anatomy, volume_4d[..., 2]), 'viewer should show frame 2'
    assert np.array_equal(viewer.data.astype(bool), mask_4d[..., 2].astype(bool))
    viewer.real_close()


if __name__ == '__main__':
    common.run_tests(globals())
