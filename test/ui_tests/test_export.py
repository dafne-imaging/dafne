"""
Saving and exporting: mask export in npz/npy/nifti/compact-nifti formats, ROI pickle
save/load round trip, statistics export, and saving the image data as NIfTI.
"""

import os
import csv
import numpy as np
import nibabel as nib

import common
from common import (create_main_window, process_events, wait_threads, temp_dir,
                    set_active_roi, make_test_volume, make_square_mask, make_bundle)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 6)
volume = make_test_volume(SHAPE)
mask = make_square_mask(SHAPE, 5, 15, 5, 15)

bundle = make_bundle(os.path.join(tmpdir, 'export.npz'), volume, {'muscle': mask})
win.loadDirectory(bundle)
process_events()


def test_export_npz():
    out_file = os.path.join(tmpdir, 'masks.npz')
    win.saveResults(out_file, 'npz', False)
    wait_threads()
    saved = np.load(out_file, allow_pickle=True)
    assert 'muscle' in saved
    assert saved['muscle'].shape == SHAPE
    assert np.array_equal(saved['muscle'].astype(bool), mask.astype(bool))


def test_export_npy():
    out_dir = os.path.join(tmpdir, 'npy_out')
    os.makedirs(out_dir, exist_ok=True)
    win.saveResults(out_dir, 'npy', False)
    wait_threads()
    saved = np.load(os.path.join(out_dir, 'muscle.npy'))
    assert np.array_equal(saved.astype(bool), mask.astype(bool))


def test_export_nifti():
    out_dir = os.path.join(tmpdir, 'nii_out')
    os.makedirs(out_dir, exist_ok=True)
    win.saveResults(out_dir, 'nifti', False)
    wait_threads()
    nii = nib.load(os.path.join(out_dir, 'muscle.nii.gz'))
    assert int((np.asarray(nii.dataobj) > 0).sum()) == int(mask.sum())


def test_export_compact_nifti():
    out_file = os.path.join(tmpdir, 'compact.nii.gz')
    win.saveResults(out_file, 'compact_nifti', False)
    wait_threads()
    nii = nib.load(out_file)
    assert int((np.asarray(nii.dataobj) > 0).sum()) == int(mask.sum())


def test_bundle_round_trip():
    out_file = os.path.join(tmpdir, 'bundle_out.npz')
    win.saveBundle(out_file, 'test bundle')
    wait_threads()
    saved = np.load(out_file, allow_pickle=True)
    assert np.allclose(saved['data'], volume)
    assert np.allclose(saved['resolution'], win.resolution[:3])
    assert np.array_equal(saved['mask_muscle'].astype(bool), mask.astype(bool))
    assert str(saved['comment']) == 'test bundle'
    # reload it and check the state is reconstructed
    win.loadDirectory(out_file)
    process_events()
    assert 'muscle' in win.roiManager.get_roi_names()
    assert np.array_equal(win.roiManager.get_mask('muscle', 1).astype(bool),
                          mask[:, :, 1].astype(bool))


def test_roi_pickle_round_trip():
    pickle_file = os.path.join(tmpdir, 'rois.p')
    win.saveROIPickle(pickle_file)
    wait_threads()
    assert os.path.exists(pickle_file)
    win.clearAllROIs()
    process_events()
    assert not win.roiManager.get_roi_names()
    win.loadROIPickle(pickle_file)
    process_events()
    assert 'muscle' in win.roiManager.get_roi_names()
    assert np.array_equal(win.roiManager.get_mask('muscle', 2).astype(bool),
                          mask[:, :, 2].astype(bool))


def test_statistics_export():
    set_active_roi(win, 'muscle')
    stats_file = os.path.join(tmpdir, 'stats.csv')
    win.saveStats(stats_file)
    wait_threads()
    assert os.path.exists(stats_file)
    with open(stats_file, newline='') as f:
        content = list(csv.reader(f))
    flat = '\n'.join(','.join(row) for row in content)
    assert 'muscle' in flat, 'statistics file should mention the ROI'


def test_reorient_data():
    # save_data_as_reoriented_nifti pops an orientation dialog, so here we exercise the
    # non-interactive reorientation slot instead
    before = win.medical_volume.volume.copy()
    win.reorient_data('Invert Slices')
    process_events()
    assert win.medical_volume.volume.shape == before.shape
    assert np.allclose(win.medical_volume.volume, np.flip(before, axis=2)), \
        'slices should be inverted'


if __name__ == '__main__':
    common.run_tests(globals())
