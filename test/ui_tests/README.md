# Non-interactive UI tests

These tests instantiate the real Dafne interface (main window, toolbox, viewers) and
drive it programmatically through the same signals, slots and widget interactions the
user would trigger — but without any user interaction, so they can run unattended.
Windows are created but kept hidden, and all blocking dialogs (alerts, questions,
confirmations) are suppressed by the harness in `common.py`.

They are *not* pytest tests: the interface cannot be torn down and rebuilt repeatedly
inside one process, so each script is standalone and `run_all.py` executes each one in
a separate process.

## Running

Run everything (from the repository root, inside the Dafne environment):

    python test/ui_tests/run_all.py

Run a subset by substring:

    python test/ui_tests/run_all.py viewer export

Run a single script directly:

    python test/ui_tests/test_roi_editing.py

A graphical environment (or a virtual one, e.g. `xvfb-run`) is required, since real Qt
windows and OpenGL (pyvista) widgets are created.

## Coverage

| Script | What it tests |
| --- | --- |
| `test_loading.py` | Loading numpy bundles (3D/4D, with masks), NIfTI, DICOM series; post-load interface state |
| `test_roi_editing.py` | ROI add/select/copy/combine/delete, mask grow/shrink/despeckle/fill-holes/clear, remove overlap, undo/redo, mask↔contour round trip |
| `test_interpolation.py` | Slice and block mask interpolation, current-ROI vs all-ROIs, toolbox interpolation-method radio buttons |
| `test_export.py` | Mask export (npz/npy/nifti/compact nifti), bundle round trip, ROI pickle round trip, statistics CSV, data-as-NIfTI export |
| `test_multicontrast.py` | Adding/switching/deleting contrasts, selector combo state, geometry mismatch rejection, contrasts in bundles |
| `test_timeresolved.py` | 4D detection, timepoint navigation, per-frame ROIs, time copy/interpolation, single-frame vs all-frames export |
| `test_toolbox.py` | Edit modes, brush controls, ROI combo sync, undo/redo/export enabling, timepoint slider, contrast combo, splash, general enable |
| `test_viewer3d.py` | Triplanar/3D viewer standalone: crosshair navigation, overlays, 3D actors, slice-sync signals |
| `test_viewer3d_integration.py` | Viewer wired to the main window: bidirectional slice sync, ROI/contrast/timepoint updates |
| `test_support_viewer.py` | ROI transfer dialog: bundle loading, per-ROI checkboxes, orientation controls, transfer signal payload |

Not covered (require heavy external components): deep-learning segmentation and model
downloads, elastix-based registration/propagation, SAM-based operations, incremental
learning, radiomics.
