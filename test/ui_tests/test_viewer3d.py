"""
Standalone triplanar/3D viewer: data feeding, crosshair navigation, mask overlays,
3D actors, main-slice synchronization signals and the other-ROIs toggle.
"""

import numpy as np

import common
from common import get_app, process_events

app = get_app()

from dafne.ui.Viewer3D import Viewer3D, MAIN_PLANE_AXIS

viewer = Viewer3D()

rng = np.random.default_rng(0)
NR, NC, NS = 40, 30, 12
anatomy = rng.uniform(20, 1000, (NR, NC, NS)).astype(np.float32)
active = np.zeros((NR, NC, NS), dtype=np.uint8)
active[10:20, 10:20, 3:8] = 1
other = np.zeros((NR, NC, NS), dtype=np.uint8)
other[25:35, 5:15, 2:9] = 2
other[2:8, 20:28, 4:10] = 3

received = []
viewer.main_slice_changed.connect(lambda n: received.append(n))

SPACING = [1.0, 1.2, 5.0]


class FakeEvent:
    pass


def get_view(fixed_axis):
    return [v for v in viewer.views if v.fixed_axis == fixed_axis][0]


def test_hidden_state_storage():
    # feeding data while hidden must not crash and must be stored
    viewer.set_spacing_and_anatomy(SPACING, anatomy)
    viewer.set_main_slice(4)
    assert viewer.position[MAIN_PLANE_AXIS] == 4
    assert viewer.position[0] == NR // 2 and viewer.position[1] == NC // 2


def test_show_and_populate():
    viewer.show()
    process_events()
    viewer.set_spacing_and_anatomy(SPACING, anatomy)
    viewer.set_main_slice(4)
    viewer.set_spacing_and_data(SPACING, active)
    viewer.set_other_masks(SPACING, other)
    process_events()

    for v in viewer.views:
        assert len(v.imList) == anatomy.shape[v.fixed_axis]
        assert int(v.curImage) == viewer.position[v.fixed_axis]
        assert v.crosshair_hline is not None and v.crosshair_vline is not None
    assert get_view(MAIN_PLANE_AXIS).image.shape == (NR, NC)
    assert get_view(1).image.shape == (NR, NS)
    assert get_view(0).image.shape == (NC, NS)


def test_mask_overlays():
    main_view = get_view(MAIN_PLANE_AXIS)
    assert main_view.maskImPlot is not None
    assert main_view.otherMaskImPlot is not None
    assert np.any(main_view.maskImPlot.get_array()), 'active mask should be drawn'


def test_3d_actors():
    assert viewer.actor_roi is not None
    assert viewer.actor_anatomy is not None
    assert len(viewer.other_actors) == 2, 'one mesh per non-active ROI'


def test_click_navigation():
    received.clear()
    main_view = get_view(MAIN_PLANE_AXIS)
    ortho1 = get_view(1)
    ortho0 = get_view(0)
    # clicking an orthogonal view changes the main slice and notifies the main window
    ev = FakeEvent()
    ev.inaxes = ortho1.axes
    ev.xdata = 7.2  # slice (axis 2) coordinate
    ev.ydata = 15.0  # row (axis 0) coordinate
    ortho1.leftPressCB(ev)
    assert viewer.position[2] == 7 and viewer.position[0] == 15
    assert received == [7]
    assert int(main_view.curImage) == 7
    assert int(ortho0.curImage) == 15

    # clicking the main view moves the crosshair without changing the main slice
    received.clear()
    ev2 = FakeEvent()
    ev2.inaxes = main_view.axes
    ev2.xdata = 12.0  # column (axis 1)
    ev2.ydata = 22.0  # row (axis 0)
    main_view.leftPressCB(ev2)
    assert viewer.position == [22, 12, 7]
    assert received == []
    assert int(ortho1.curImage) == 12
    assert int(ortho0.curImage) == 22


def test_set_main_slice_does_not_notify():
    received.clear()
    viewer.set_main_slice(9)
    assert viewer.position[2] == 9
    assert int(get_view(MAIN_PLANE_AXIS).curImage) == 9
    assert received == []


def test_position_clamping():
    viewer.set_position({0: 999, 1: -5})
    assert viewer.position[0] == NR - 1 and viewer.position[1] == 0


def test_set_slice():
    new_slice = np.zeros((NR, NC), dtype=np.uint8)
    new_slice[0:5, 0:5] = 1
    viewer.set_slice(9, new_slice)
    assert np.array_equal(np.asarray(get_view(MAIN_PLANE_AXIS).maskImPlot.get_array()),
                          new_slice)


def test_toggle_other_rois():
    viewer.show_other_rois_checkbox.setChecked(False)
    process_events()
    assert len(viewer.other_actors) == 0
    assert not np.any(np.asarray(get_view(MAIN_PLANE_AXIS).otherMaskImPlot.get_array()))
    viewer.show_other_rois_checkbox.setChecked(True)
    process_events()
    assert len(viewer.other_actors) == 2


def test_scroll_notifies_main():
    received.clear()
    main_view = get_view(MAIN_PLANE_AXIS)
    ev = FakeEvent()
    ev.inaxes = main_view.axes
    ev.x, ev.y = 1, 1
    ev.step = -1
    main_view.last_scroll_time = 0
    main_view.mouseScrollCB(ev)
    assert received == [int(main_view.curImage)]


def test_shape_change_resets_position():
    anatomy2 = np.random.default_rng(1).uniform(20, 1000, (20, 20, 5)).astype(np.float32)
    viewer.set_spacing_and_anatomy([1.0, 1.0, 1.0], anatomy2)
    assert viewer.position == [10, 10, 4]
    viewer.real_close()


if __name__ == '__main__':
    common.run_tests(globals())
