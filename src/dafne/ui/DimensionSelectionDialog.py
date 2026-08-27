"""
Dialog to let the user choose, for a dataset with more than 4 dimensions,
which axes are spatial, which one (if any) is time, and which fixed index
to use for any other axis, so that the array can be reduced to a plain
3D or 3D+time volume before it is loaded.
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                              QSpinBox, QPushButton, QMessageBox, QScrollArea, QWidget)

ROLE_SPATIAL = 'Spatial'
ROLE_TIME = 'Time'
ROLE_FIXED = 'Fixed (pick index)'


class DimensionSelectionDialog(QDialog):

    def __init__(self, shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select dimensions to load')
        self.shape = shape
        self.role_combos = []
        self.index_spins = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            'This dataset has {} dimensions, with shape {}.\n'
            'Choose exactly 3 spatial dimensions, optionally one time dimension,\n'
            'and a fixed index for any other dimension.'.format(len(shape), tuple(shape))))

        scroll = QScrollArea()
        scroll_widget = QWidget()
        form_layout = QVBoxLayout(scroll_widget)

        for axis, size in enumerate(shape):
            row = QHBoxLayout()
            row.addWidget(QLabel('Axis {} (size {}):'.format(axis, size)))

            combo = QComboBox()
            combo.addItems([ROLE_SPATIAL, ROLE_TIME, ROLE_FIXED])
            if axis < 3:
                combo.setCurrentText(ROLE_SPATIAL)
            elif axis == 3:
                combo.setCurrentText(ROLE_TIME)
            else:
                combo.setCurrentText(ROLE_FIXED)
            row.addWidget(combo)

            spin = QSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(max(size - 1, 0))
            spin.setValue(0)
            spin.setEnabled(combo.currentText() == ROLE_FIXED)
            row.addWidget(spin)

            combo.currentTextChanged.connect(lambda text, s=spin: s.setEnabled(text == ROLE_FIXED))

            form_layout.addLayout(row)
            self.role_combos.append(combo)
            self.index_spins.append(spin)

        scroll_widget.setLayout(form_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        button_row = QHBoxLayout()
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.validate_and_accept)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

    def validate_and_accept(self):
        roles = [combo.currentText() for combo in self.role_combos]
        if roles.count(ROLE_SPATIAL) != 3:
            QMessageBox.warning(self, 'Invalid selection', 'Please select exactly 3 spatial dimensions.')
            return
        if roles.count(ROLE_TIME) > 1:
            QMessageBox.warning(self, 'Invalid selection', 'Please select at most one time dimension.')
            return
        self.accept()

    def get_selection(self):
        """ Returns (roles, fixed_indices): roles is a list with one of
            ROLE_SPATIAL/ROLE_TIME/ROLE_FIXED per axis; fixed_indices has the
            chosen index for axes marked ROLE_FIXED (None otherwise). """
        roles = [combo.currentText() for combo in self.role_combos]
        fixed_indices = [spin.value() if role == ROLE_FIXED else None
                          for role, spin in zip(roles, self.index_spins)]
        return roles, fixed_indices


def reduce_array_dimensions(data, parent=None):
    """ If data has more than 4 dimensions, ask the user which axes are spatial,
        which one (if any) is time, and which index to keep for the remaining
        (fixed) axes, then return the reduced array (3D, or 4D with time last).
        Returns the array unchanged if it already has 4 or fewer dimensions.
        Returns None if the user cancels the dialog. """
    if data.ndim <= 4:
        return data

    dialog = DimensionSelectionDialog(data.shape, parent)
    if dialog.exec_() != QDialog.Accepted:
        return None

    roles, fixed_indices = dialog.get_selection()

    reduced = data
    # remove fixed axes from the highest index down, so lower axis indices stay valid
    for axis in reversed(range(data.ndim)):
        if roles[axis] == ROLE_FIXED:
            reduced = np.take(reduced, fixed_indices[axis], axis=axis)

    kept_axes = [axis for axis in range(data.ndim) if roles[axis] != ROLE_FIXED]
    kept_roles = [roles[axis] for axis in kept_axes]
    spatial_positions = [i for i, role in enumerate(kept_roles) if role == ROLE_SPATIAL]
    time_positions = [i for i, role in enumerate(kept_roles) if role == ROLE_TIME]
    reduced = np.transpose(reduced, spatial_positions + time_positions)
    return np.ascontiguousarray(reduced)
