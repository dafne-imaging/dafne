"""
Multicontrast support: adding, switching, deleting additional contrasts, the toolbox
contrast selector state, geometry checks, and inclusion in exported bundles.
"""

import os
import numpy as np

import common
from common import (create_main_window, process_events, wait_threads, temp_dir,
                    make_test_volume, make_square_mask, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 6)
volume = make_test_volume(SHAPE)
extra = np.full(SHAPE, 500.0, dtype=np.float32)

bundle = make_bundle(os.path.join(tmpdir, 'base.npz'), volume,
                     {'muscle': make_square_mask(SHAPE, 5, 15, 5, 15)})
extra_file = make_bundle(os.path.join(tmpdir, 'extra.npz'), extra)
bad_file = make_bundle(os.path.join(tmpdir, 'bad.npz'),
                       make_test_volume((10, 10, 2), seed=2))

win.loadDirectory(bundle)
process_events()

from dafne.ui.ToolboxWindow import ToolboxWindow


def test_add_contrast():
    win._load_additional_contrast_data(extra_file, 'extra')
    wait_threads()
    assert 'extra' in win.additional_contrasts
    assert tb.find_contrast_in_combo('extra') >= 0
    assert tb.find_contrast_in_combo(ToolboxWindow.BASE_CONTRAST_LABEL) >= 0
    assert tb.contrastFrame.isVisibleTo(tb), 'contrast selector should appear'
    assert np.allclose(win.additional_contrasts['extra'].volume, extra)


def test_switch_contrast():
    win.change_contrast('extra')
    process_events()
    assert win.current_contrast == 'extra'
    assert np.allclose(np.asarray(win.image), extra[:, :, int(win.curImage)], atol=1e-3)
    # ROIs are shared between contrasts
    assert 'muscle' in win.roiManager.get_roi_names()
    win.change_contrast(ToolboxWindow.BASE_CONTRAST_LABEL)
    process_events()
    assert np.allclose(np.asarray(win.image), volume[:, :, int(win.curImage)], atol=1e-3)


def test_mismatched_contrast_resampled():
    # the small dataset carries valid geometry, so it is resampled onto the base grid
    win._load_additional_contrast_data(bad_file, 'bad')
    wait_threads()
    assert 'bad' in win.additional_contrasts
    assert win.additional_contrasts['bad'].volume.shape == SHAPE, \
        'dataset with valid geometry should be resampled onto the base grid'
    win.delete_additional_contrast('bad')
    process_events()


def test_contrast_in_bundle():
    out_file = os.path.join(tmpdir, 'bundle_out.npz')
    win.saveBundle(out_file, 'contrast bundle')
    wait_threads()
    saved = np.load(out_file, allow_pickle=True)
    assert [str(n) for n in saved['contrast_names']] == ['extra']
    assert np.allclose(saved['data2'], extra)


def test_delete_contrast():
    win.delete_additional_contrast('extra')
    process_events()
    assert 'extra' not in win.additional_contrasts
    assert tb.find_contrast_in_combo('extra') < 0
    # base contrast is still the displayed one
    assert np.allclose(np.asarray(win.image), volume[:, :, int(win.curImage)], atol=1e-3)


if __name__ == '__main__':
    common.run_tests(globals())
