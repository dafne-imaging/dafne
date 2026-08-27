"""
Support data viewer (ROI transfer dialog): loading a numpy bundle, per-ROI checkboxes,
orientation controls (transpose/mirror/rotate) and the transfer signal payload.
"""

import os
import numpy as np

import common
from common import get_app, process_events, temp_dir, make_test_volume, make_bundle

app = get_app()
tmpdir = temp_dir()

from PyQt5 import QtWidgets
from dafne.ui.SupportDataViewer import SupportDataViewerDialog

SHAPE = (24, 32, 5)
volume = make_test_volume(SHAPE)
mask_one = np.zeros(SHAPE, dtype=np.uint8)
mask_one[2:10, 2:10, :] = 1
mask_two = np.zeros(SHAPE, dtype=np.uint8)
mask_two[12:20, 12:22, :] = 1

bundle = make_bundle(os.path.join(tmpdir, 'support.npz'), volume,
                     {'one': mask_one, 'two': mask_two}, resolution=[1.0, 1.5, 5.0])

dialog = SupportDataViewerDialog(bundle)
process_events()


def get_checkboxes():
    return [w for w in dialog.roiGroup.findChildren(QtWidgets.QCheckBox)]


def test_dialog_loads_bundle():
    assert dialog.imshowWidget.data.shape == SHAPE
    assert set(dialog.imshowWidget.masks.keys()) == {'one', 'two'}
    boxes = get_checkboxes()
    assert sorted(box.text() for box in boxes) == ['one', 'two']
    assert all(box.isChecked() for box in boxes)


def test_orientation_controls():
    viewer = dialog.imshowWidget
    viewer.transpose_all()
    process_events()
    assert viewer.data.shape == (SHAPE[1], SHAPE[2], SHAPE[0])
    assert viewer.masks['one'].shape == viewer.data.shape
    assert list(viewer.resolution) == [1.5, 5.0, 1.0]
    # two more transpositions bring it back
    viewer.transpose_all()
    viewer.transpose_all()
    process_events()
    assert viewer.data.shape == SHAPE
    assert np.allclose(viewer.data, volume)

    viewer.mirror_x_all()
    process_events()
    assert np.allclose(viewer.data, np.flip(volume, axis=1))
    assert np.array_equal(viewer.masks['one'], np.flip(mask_one, axis=1))
    viewer.mirror_x_all()  # back

    viewer.mirror_y_all()
    process_events()
    assert np.allclose(viewer.data, np.flip(volume, axis=0))
    viewer.mirror_y_all()  # back

    viewer.rotate_90_all()
    process_events()
    assert viewer.data.shape == (SHAPE[1], SHAPE[0], SHAPE[2])
    for _ in range(3):
        viewer.rotate_90_all()
    process_events()
    assert np.allclose(viewer.data, volume)
    assert np.array_equal(viewer.masks['two'], mask_two)


def test_mask_toggle_and_transfer_signal():
    received = []
    dialog.mask_transfer_signal.connect(
        lambda data, masks, resolution: received.append((data, masks, resolution)))

    # disable one mask through its checkbox, as the user would
    box = [b for b in get_checkboxes() if b.text() == 'one'][0]
    box.setChecked(False)
    box.clicked.emit(False)
    process_events()
    assert dialog.imshowWidget.enabled_masks['one'] is False

    dialog.emit_signal()
    assert len(received) == 1
    data, masks, resolution = received[0]
    assert data.shape == SHAPE
    assert set(masks.keys()) == {'two'}, 'only the enabled masks must be transferred'
    assert np.array_equal(masks['two'], mask_two)
    assert len(resolution) == 3


if __name__ == '__main__':
    common.run_tests(globals())
