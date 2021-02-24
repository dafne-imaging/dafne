#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any, Union
from collections import OrderedDict
from PyQt5.QtWidgets import QDialog, QWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QFormLayout, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QDialogButtonBox, QLabel, QApplication, QSlider, QCheckBox, QComboBox, QSizePolicy, \
    QTabWidget
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
import abc
import sys
import math


class MixedDict(OrderedDict):
    """
    A class that behaves as a list and as a dictionary.
    If the key is integer, returns the item at that position, otherwise behave as a dictionary
    """

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

    def key_at(self, pos):
        return list(self.keys())[pos]

    def __getitem__(self, key):
        if type(key) == int:
            return OrderedDict.__getitem__(self, self.key_at(key))
        else:
            return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if type(key) == int:
            dict_key = self.key_at(key)  # if this fails, it will raise a keyerror, and we are fine with that
        else:
            dict_key = key
        OrderedDict.__setitem__(self, dict_key, value)

    def __iter__(self):
        return iter(self.values())


class InputClass(abc.ABC):
    """
    Abstract implementation of an input class
    """

    @abc.abstractmethod
    def get_widget(self) -> QWidget:
        pass

    @abc.abstractmethod
    def get_label(self) -> str:
        pass

    @abc.abstractmethod
    def get_value(self) -> Any:
        pass


## Default input classes

class TextLineInput(InputClass):

    def __init__(self, label:str, initial_value:str =''):
        self.label = label
        self.line_edit = QLineEdit()
        self.line_edit.setText(initial_value)

    def get_widget(self) -> QWidget:
        return self.line_edit

    def get_label(self) -> str:
        return self.label

    def get_value(self) -> Any:
        return self.line_edit.text()


class IntSpinInput(InputClass):

    def __init__(self, label:str, initial_value:int=0, min:int=0, max:int=99, increment:int=1):
        self.label = label
        self.widget = QSpinBox()
        self.widget.setMinimum(min)
        self.widget.setMaximum(max)
        self.widget.setValue(initial_value)
        self.widget.setSingleStep(increment)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.value()


class FloatSpinInput(InputClass):

    def __init__(self, label:str, initial_value:float=0.0, min:float=0.0, max:float=99.0, increment:float=1.0):
        self.label = label
        self.widget = QDoubleSpinBox()
        self.widget.setMinimum(min)
        self.widget.setMaximum(max)
        self.widget.setValue(initial_value)
        self.widget.setSingleStep(increment)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.value()


class IntSliderInput(InputClass):

    def __init__(self, label:str, initial_value:int=0, min:int=0, max:int=99, increment:int=1):
        self.label = label
        self.widget = QWidget()
        self.slider = QSlider(self.widget)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(min)
        self.slider.setMaximum(max)
        self.slider.setValue(initial_value)
        self.slider.setSingleStep(increment)
        self.slider_label = QLabel(self.widget)
        self.slider_label.setText(str(initial_value))
        layout = QHBoxLayout(self.widget)
        layout.addWidget(self.slider)
        layout.addWidget(self.slider_label)
        self.slider.valueChanged.connect(lambda value: self.slider_label.setText(str(value)))

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.slider.value()

class BooleanInput(InputClass):

    def __init__(self, label: str, default:bool = False):
        self.label = label
        self.widget = QCheckBox()
        self.widget.setChecked(default)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.isChecked()


class FloatSliderWidget(QWidget):

    valueChanged = pyqtSignal(float)

    def __init__(self, minimum_value: float = 0.0, maximum_value: float = 1.0, increment: Union[float, None] = None,
                 initial_value: float = 0, parent=None):
        QWidget.__init__(self, parent)
        if increment is None:
            increment = (maximum_value - minimum_value) / 100

        self.min = minimum_value
        self.max = maximum_value
        self.increment = increment

        # set the significant digits displayed so that the increment is always displayed
        self.significant_digits = int(math.log10(max([abs(self.min), abs(self.max)]) / self.increment)) + 1

        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.value_to_slider(self.max))
        self.slider.setValue(self.value_to_slider(initial_value))
        self.slider.setSingleStep(1)
        self.slider_label = QLabel(self)
        self.slider_label.setText(self.format_label(initial_value))
        layout = QHBoxLayout(self)
        layout.addWidget(self.slider)
        layout.addWidget(self.slider_label)
        self.slider.valueChanged.connect(lambda value: self.slider_label.setText(
            self.format_label(self.slider_to_value(value))))
        self.slider.valueChanged.connect(lambda value: self.valueChanged.emit(self.slider_to_value(value)))

    # convert an int value of the slider position to actual value
    def slider_to_value(self, slider_value):
        return slider_value * self.increment + self.min

    def value_to_slider(self, value):
        return int((value - self.min) / self.increment)

    def format_label(self, value):
        return ('{:' + str(self.significant_digits) + 'g}').format(value)

    def setValue(self, value):
        self.slider.setValue(self.value_to_slider(value))

    def value(self):
        return self.slider_to_value(self.slider.value())


class FloatSliderInput(InputClass):

    def __init__(self, label: str, default: float = 0, minimum_value: float = 0.0, maximum_value: float = 1.0, increment: Union[float, None] = None):
        self.label = label
        self.widget = FloatSliderWidget(minimum_value, maximum_value, increment, default)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.value()


class ColorSliderInput(InputClass):
    def __init__(self, label: str, default_color: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.label = label
        self.widget = QWidget()
        self.slider_container = QWidget()
        self.color_label = QLabel()

        self.color_label.setMinimumWidth(30)

        self.red_slider = FloatSliderWidget(0.0, 1.0, 0.1, default_color[0])
        self.green_slider = FloatSliderWidget(0.0, 1.0, 0.1, default_color[1])
        self.blue_slider = FloatSliderWidget(0.0, 1.0, 0.1, default_color[2])

        self._update_label_color()
        vlayout = QVBoxLayout(self.slider_container)
        vlayout.setContentsMargins(0,0,0,0)
        vlayout.addWidget(self.red_slider)
        vlayout.addWidget(self.green_slider)
        vlayout.addWidget(self.blue_slider)
        hlayout = QHBoxLayout(self.widget)
        hlayout.addWidget(self.slider_container)
        hlayout.addWidget(self.color_label)

        self.red_slider.valueChanged.connect(lambda value: self._update_label_color())
        self.green_slider.valueChanged.connect(lambda value: self._update_label_color())
        self.blue_slider.valueChanged.connect(lambda value: self._update_label_color())

    @pyqtSlot()
    def _update_label_color(self):
        color = self.get_value()
        color_string = 'background-color: #{:02x}{:02x}{:02x};'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        self.color_label.setStyleSheet(color_string)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return (self.red_slider.value(), self.green_slider.value(), self.blue_slider.value())

class ColorSpinInput(InputClass):
    def __init__(self, label: str, default_color: Union[tuple[float, float, float], tuple[float, float, float, float]] = (1.0, 1.0, 1.0), has_alpha = False):
        self.label = label
        self.widget = QWidget()
        self.color_label = QLabel()

        if has_alpha or len(default_color) == 4:
            self.has_alpha = True
        else:
            self.has_alpha = False

        if len(default_color) == 3: # add alpha channel
            default_color = (*default_color, 1.0)

        self.color_label.setMinimumWidth(30)

        self.red_spin = QDoubleSpinBox()
        self.red_spin.setMinimum(0.0)
        self.red_spin.setMaximum(1.0)
        self.red_spin.setSingleStep(0.1)
        self.red_spin.setValue(default_color[0])

        self.green_spin = QDoubleSpinBox()
        self.green_spin.setMinimum(0.0)
        self.green_spin.setMaximum(1.0)
        self.green_spin.setSingleStep(0.1)
        self.green_spin.setValue(default_color[1])

        self.blue_spin = QDoubleSpinBox()
        self.blue_spin.setMinimum(0.0)
        self.blue_spin.setMaximum(1.0)
        self.blue_spin.setSingleStep(0.1)
        self.blue_spin.setValue(default_color[2])

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setMinimum(0.0)
        self.alpha_spin.setMaximum(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(default_color[3])

        self._update_label_color()
        hlayout = QHBoxLayout(self.widget)
        hlayout.addWidget(self.red_spin)
        hlayout.addWidget(self.green_spin)
        hlayout.addWidget(self.blue_spin)

        if self.has_alpha:
            hlayout.addWidget(self.alpha_spin)

        hlayout.addWidget(self.color_label)

        self.red_spin.valueChanged.connect(lambda value: self._update_label_color())
        self.green_spin.valueChanged.connect(lambda value: self._update_label_color())
        self.blue_spin.valueChanged.connect(lambda value: self._update_label_color())
        self.alpha_spin.valueChanged.connect(lambda value: self._update_label_color())

    @pyqtSlot()
    def _update_label_color(self):
        color = self.get_value(force_alpha=True)
        color_string = 'background-color: rgba({},{},{},{});'.format(int(color[0]*255),
                                                                     int(color[1]*255),
                                                                     int(color[2]*255),
                                                                     int(color[3]*255))
        self.color_label.setStyleSheet(color_string)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self, force_alpha = False) -> Any:
        if self.has_alpha or force_alpha:
            return (self.red_spin.value(), self.green_spin.value(), self.blue_spin.value(), self.alpha_spin.value())
        else:
            return (self.red_spin.value(), self.green_spin.value(), self.blue_spin.value())

class OptionInput(InputClass):

    def __init__(self, label:str, value_list:list[Union[str, tuple[str, Any]]], default=None):
        self.label = label
        self.widget = QComboBox()

        self.output_list = []
        default_set = False

        for index, v in enumerate(value_list):
            if type(v) == str:
                self.widget.addItem(v)
                self.output_list.append(v)
                if default is not None:
                    if default == v:
                        default_set = True
                        self.widget.setCurrentIndex(index)
            else: # it is a tuple, specified by the type hint in the constructor
                self.widget.addItem(v[0])
                self.output_list.append(v[1])
                if default is not None:
                    if default in v:
                        default_set = True
                        self.widget.setCurrentIndex(index)

        if not default_set and type(default) == int:
            self.widget.setCurrentIndex(default)

    def add_option(self, option: Union[str, tuple[str, Any]]):
        if type(option) == str:
            self.widget.addItem(option)
            self.output_list.append(option)
        else:  # it is a tuple, specified by the type hint in the constructor
            self.widget.addItem(option[0])
            self.output_list.append(option[1])

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.output_list[self.widget.currentIndex()]

class PageWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.formLayout = QFormLayout(self)
        self.current_position = 0

    def add_object(self, label_text, widget):
        label = QLabel(self)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setText(label_text)
        self.formLayout.setWidget(self.current_position, QFormLayout.LabelRole, label)
        self.formLayout.setWidget(self.current_position, QFormLayout.FieldRole, widget)
        self.current_position += 1


## Dialog class implementation

class GenericDialog(QDialog):

    def __init__(self, title: str, input_list: list[InputClass], entries_per_page: int=10, parent=None, message=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.verticalLayout = QVBoxLayout(self)
        self.tabWidget = QTabWidget(self)

        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(lambda: self.exit_dialog(True))
        self.buttonBox.rejected.connect(lambda: self.exit_dialog(False))

        self.input_list = input_list

        current_page_number = 1
        current_page = PageWidget(self)
        if len(input_list) <= entries_per_page:
            self.tabWidget.addTab(current_page, title)
        else:
            self.tabWidget.addTab(current_page, f'{title}: 1')

        current_entry_number = 0
        for input_obj in input_list:
            current_entry_number += 1
            if current_entry_number > entries_per_page:
                current_page = PageWidget(self)
                current_page_number += 1
                self.tabWidget.addTab(current_page, f'{title}: {current_page_number}')
                current_entry_number = 1
            current_page.add_object(input_obj.get_label(), input_obj.get_widget())

        if message:
            messageLabel = QLabel()
            messageLabel.setText(message)
            messageLabel.setWordWrap(True)
            self.verticalLayout.addWidget(messageLabel)


        self.verticalLayout.addWidget(self.tabWidget)
        self.verticalLayout.addWidget(self.buttonBox)

        self.resize(self.verticalLayout.sizeHint())

        self.output = None
        self.accepted = False
        self.setModal(True)

    def exit_dialog(self, accepted):
        self.accepted = accepted
        self.close()

    def closeEvent(self, event):
        self.output = MixedDict()
        for input_obj in self.input_list:
            self.output[input_obj.get_label()] = input_obj.get_value()


## This is the main function that should be called
def show_dialog(title: str, input_list: list[InputClass], parent=None, entries_per_page=10, message=None) -> (bool, MixedDict):
    dialog = GenericDialog(title, input_list, entries_per_page, parent, message)
    dialog.exec()
    # this will stop until the dialog is closed
    if dialog.accepted is None:
        return None
    return dialog.accepted, dialog.output


## Testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    accepted, values = show_dialog("Test", [TextLineInput('Text input'),
                                            IntSpinInput('My Int', 10, -100, 100),
                                            FloatSpinInput('My Float'),
                                            IntSliderInput('My slider'),
                                            BooleanInput('Bool value'),
                                            OptionInput('My string options', [
                                                'option 1',
                                                'option 2',
                                                'option 3'
                                            ], 'option 3'),
                                            OptionInput('My int options', [
                                                            ('option 1', 1.1),
                                                            ('option 2', 2.2),
                                                            ('option 3', 3.3)
                                                        ], 2.2),
                                            FloatSliderInput('Float slider', 0.0, 0.0, 1.0, 0.1),
                                            ColorSpinInput('Choose color', (0.5, 0.0, 1.0))
                                            ])
    # Note: for option inputs, the value list can be a list of strings, and then the output is the string itself, or a
    # list of tuples, where the first element is a string (the label) and the second is the returned value (any).
    # The default value for options can be the label string, the default returned value, or an integer index


    # returned values can be accessed by key or by position
    print(values['My Int'])
    print(values[2])

    # they can be iterated like a list
    for v in values:
        print(v)

    # or items can be accessed like a dictionary
    for k,v in values.items():
        print(k,v)
