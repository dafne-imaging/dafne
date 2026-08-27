"""
Shared harness for the non-interactive UI tests.

These tests instantiate the real Dafne interface (MuscleSegmentation + ToolboxWindow)
and drive it programmatically: no user interaction is required, but the actual
signal/slot wiring, toolbox state and viewers are exercised.

Each test script is standalone (python test/ui_tests/test_xxx.py); run_all.py runs
every script in a separate process, since the interface is not designed to be torn
down and rebuilt repeatedly within one process.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np


def get_app():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def process_events():
    get_app().processEvents()


def wait_threads():
    """ Wait for all @separate_thread_decorator background jobs to finish. """
    from dafne.utils.ThreadHelpers import threadpool
    threadpool.waitForDone()
    process_events()


def create_main_window():
    """ Create a full Dafne interface (main window + toolbox), hidden and with all
        blocking dialogs suppressed. Returns (window, toolbox, alerts) where alerts
        collects the messages that would have been shown to the user. """
    get_app()
    from dafne.ui.MuscleSegmentation import MuscleSegmentation
    from dafne.config import GlobalConfig

    GlobalConfig['DO_INCREMENTAL_LEARNING'] = False

    win = MuscleSegmentation()
    sys.excepthook = sys.__excepthook__  # Dafne installs a dialog-based excepthook

    alerts = []

    def fake_alert(text, alert_type='Warning'):
        print('[ALERT]', text)
        alerts.append(text)

    win.alert = fake_alert
    try:
        win.alert_signal.disconnect()
    except TypeError:
        pass
    win.question = lambda *args, **kwargs: False

    tb = win.toolbox_window
    tb.hide()
    try:
        win.fig.canvas.manager.window.hide()
    except Exception:
        pass
    return win, tb, alerts


def temp_dir():
    return tempfile.mkdtemp(prefix='dafne_uitest_')


def set_active_roi(win, name, subroi=0):
    """ Select a ROI as the toolbox would when the user picks it from the combo. """
    win.toolbox_window.set_current_roi(name, subroi)
    win.changeRoi(name, subroi)
    process_events()


### test dataset factories ###############################################################

DEFAULT_RESOLUTION = np.array([1.5, 1.5, 5.0])


def make_test_volume(shape=(32, 32, 6), seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(20, 1000, shape).astype(np.float32)


def make_square_mask(shape, r0, r1, c0, c1, slices=None):
    mask = np.zeros(shape, dtype=np.uint8)
    if slices is None:
        mask[r0:r1, c0:c1, ...] = 1
    else:
        for sl in slices:
            mask[r0:r1, c0:c1, sl] = 1
    return mask


def make_bundle(path, data, masks=None, resolution=DEFAULT_RESOLUTION):
    """ Write a Dafne numpy bundle (.npz) with optional masks. """
    contents = {'data': data, 'resolution': np.asarray(resolution)}
    for name, mask in (masks or {}).items():
        contents['mask_' + name] = mask
    np.savez(path, **contents)
    return path


def make_nifti(path, data, resolution=DEFAULT_RESOLUTION):
    import nibabel as nib
    affine = np.diag([resolution[0], resolution[1], resolution[2], 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)
    return path


def make_dicom_series(directory, data, resolution=DEFAULT_RESOLUTION):
    """ Write one classic MR DICOM file per slice of a 3D volume. """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian

    os.makedirs(directory, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()
    scaled = data.astype(np.float64)
    scaled = (scaled - scaled.min()) / max(scaled.max() - scaled.min(), 1e-9) * 4000
    volume = scaled.astype(np.uint16)

    for sl in range(volume.shape[2]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = 'MR'
        ds.PatientName = 'Dafne^Test'
        ds.PatientID = 'DAFNETEST'
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SeriesNumber = 1
        ds.InstanceNumber = sl + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(sl) * float(resolution[2])]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [float(resolution[0]), float(resolution[1])]
        ds.SliceThickness = float(resolution[2])
        ds.SpacingBetweenSlices = float(resolution[2])
        ds.SliceLocation = float(sl) * float(resolution[2])
        ds.Rows, ds.Columns = volume.shape[0], volume.shape[1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = volume[:, :, sl].tobytes()
        ds.save_as(os.path.join(directory, 'slice_{:03d}.dcm'.format(sl)),
                   write_like_original=False)
    return directory


### tiny test runner #####################################################################

def run_tests(namespace):
    """ Run every test_* function defined in the given namespace (module globals),
        in definition order. Exits with a nonzero code on the first failure. """
    tests = [(name, fn) for name, fn in namespace.items()
             if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        print('--- {} ---'.format(name))
        try:
            fn()
        except Exception:
            traceback.print_exc()
            print('FAILED: {}'.format(name))
            sys.exit(1)
        print('OK: {}'.format(name))
    print('ALL TESTS PASSED ({})'.format(len(tests)))
