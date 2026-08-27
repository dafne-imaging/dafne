"""
Time-resolved (4D) datasets: frame detection, timepoint navigation through the toolbox,
per-frame ROI managers, copying masks between frames, time interpolation and
single-frame vs all-frames export.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, wait_threads, temp_dir,
                    set_active_roi, make_test_volume, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 4)
NT = 4
volume_4d = make_test_volume(SHAPE + (NT,))
mask_4d = np.zeros(SHAPE + (NT,), dtype=np.uint8)
for t in range(NT):
    mask_4d[5 + 2 * t:15 + 2 * t, 5:15, :, t] = 1

bundle = make_bundle(os.path.join(tmpdir, 'time.npz'), volume_4d, {'muscle': mask_4d})
win.loadDirectory(bundle)
process_events()


def test_time_dimension_detected():
    assert win.has_time_dimension()
    assert win.n_timepoints == NT
    assert len(win.roiManagers) == NT
    # single-frame export menu only makes sense for 4D data
    assert tb.menuSave_masks_single.menuAction().isVisible()
    # per-frame masks were loaded
    for t in (0, NT - 1):
        assert np.array_equal(win.roiManagers[t].get_mask('muscle', 1).astype(bool),
                              mask_4d[:, :, 1, t].astype(bool))


def test_timepoint_navigation():
    tb.set_current_timepoint(2)
    process_events()
    assert win.current_timepoint == 2
    assert tb.get_current_timepoint() == 2
    assert np.allclose(win.medical_volume.volume, volume_4d[..., 2])
    assert win.roiManager is win.roiManagers[2]
    assert np.allclose(np.asarray(win.image), volume_4d[:, :, int(win.curImage), 2], atol=1e-3)
    win.next_timepoint()
    process_events()
    assert win.current_timepoint == 3
    win.previous_timepoint()
    process_events()
    assert win.current_timepoint == 2


def test_time_copy():
    set_active_roi(win, 'muscle')
    tb.set_current_timepoint(0)
    process_events()
    # overwrite frame 1 with an empty manager state for the copy test
    extra_mask = np.zeros(SHAPE[:2], dtype=np.uint8)
    extra_mask[20:28, 20:28] = 1
    win.roiManagers[0].set_mask('muscle', 0, extra_mask)
    win.time_copy(1, False)
    process_events()
    assert win.current_timepoint == 1, 'time_copy should move to the target frame'
    assert np.array_equal(win.roiManagers[1].get_mask('muscle', 0).astype(bool),
                          extra_mask.astype(bool))


def test_single_frame_export():
    tb.set_current_timepoint(2)
    process_events()
    out_single = os.path.join(tmpdir, 'single.npz')
    win.saveResults(out_single, 'npz', True)
    wait_threads()
    saved = np.load(out_single, allow_pickle=True)
    assert saved['muscle'].shape == SHAPE
    assert np.array_equal(saved['muscle'].astype(bool), mask_4d[..., 2].astype(bool))


def test_all_frames_export():
    out_all = os.path.join(tmpdir, 'all.npz')
    win.saveResults(out_all, 'npz', False)
    wait_threads()
    saved = np.load(out_all, allow_pickle=True)
    assert saved['muscle'].shape == SHAPE + (NT,)


def test_time_interpolate():
    # empty a middle frame and interpolate it back from the neighbors
    from dafne.ui.ToolboxWindow import ToolboxWindow
    slice_number = 1
    tb.set_current_timepoint(2)
    process_events()
    win.roiManagers[2].set_mask('muscle', slice_number, np.zeros(SHAPE[:2], dtype=np.uint8))
    win.displayImage(slice_number, redraw=False)
    win.redraw()
    set_active_roi(win, 'muscle')
    win.time_interpolate(ToolboxWindow.INTERPOLATE_MASK_INTERPOLATE, False)
    wait_threads()
    restored = win.roiManagers[2].get_mask('muscle', slice_number)
    assert np.any(restored), 'time interpolation should fill the emptied frame'


if __name__ == '__main__':
    common.run_tests(globals())
