"""
Data loading: numpy bundles (3D and 4D, with and without masks), NIfTI files and
DICOM series, and the interface state after loading.
"""

import os
import numpy as np

import common
from common import (create_main_window, wait_threads, process_events, temp_dir,
                    make_test_volume, make_square_mask, make_bundle, make_nifti,
                    make_dicom_series, DEFAULT_RESOLUTION)

win, tb, alerts = create_main_window()
tmpdir = temp_dir()

SHAPE = (32, 32, 6)
volume = make_test_volume(SHAPE)


def test_load_npz_bundle_with_masks():
    mask = make_square_mask(SHAPE, 5, 15, 5, 15)
    bundle = make_bundle(os.path.join(tmpdir, 'basic.npz'), volume, {'muscle': mask})
    win.loadDirectory(bundle)
    process_events()
    assert len(win.imList) == SHAPE[2]
    assert win.image.shape == SHAPE[:2]
    assert np.allclose(win.resolution[:3], DEFAULT_RESOLUTION)
    assert not win.has_time_dimension()
    assert win.roiManager is not None and 'muscle' in win.roiManager.get_roi_names()
    loaded_mask = win.roiManager.get_mask('muscle', 2)
    assert np.array_equal(loaded_mask.astype(bool), mask[:, :, 2].astype(bool))
    # exports must be enabled coherently with the data (no dicom headers here)
    assert tb.menuSave_as_Numpy.isEnabled()


def test_load_4d_bundle():
    nt = 3
    volume_4d = make_test_volume(SHAPE + (nt,), seed=1)
    bundle = make_bundle(os.path.join(tmpdir, 'timeresolved.npz'), volume_4d)
    win.loadDirectory(bundle)
    process_events()
    assert win.has_time_dimension()
    assert win.n_timepoints == nt
    assert len(win.time_frames) == nt
    assert np.allclose(win.medical_volume.volume, volume_4d[..., 0])
    assert len(win.roiManagers) == nt


def test_load_nifti():
    # NIfTI loading asks the user for the display orientation: stub the dialog out
    import dicomUtils.misc as dicom_utils_misc
    dicom_utils_misc.reorient_data_ui = \
        lambda medical_volume, parent_qobject=None, inplace=False: medical_volume
    nifti = make_nifti(os.path.join(tmpdir, 'data.nii.gz'), volume)
    win.loadDirectory(nifti)
    process_events()
    assert len(win.imList) == SHAPE[2]
    assert win.affine is not None
    assert np.allclose(np.abs(np.diag(win.affine)[:3]), DEFAULT_RESOLUTION)
    assert win.roiManager is not None and not win.roiManager.get_roi_names()


def test_load_dicom_series():
    dicom_dir = make_dicom_series(os.path.join(tmpdir, 'dicoms'), volume)
    win.loadDirectory(dicom_dir)
    process_events()
    assert len(win.imList) == SHAPE[2]
    assert win.dicomHeaderList is not None
    assert win.resolution_valid
    assert np.allclose(win.resolution[:3], DEFAULT_RESOLUTION)
    # dicom data allows dicom mask export
    assert tb.actionSave_as_Dicom.isEnabled()


if __name__ == '__main__':
    common.run_tests(globals())
