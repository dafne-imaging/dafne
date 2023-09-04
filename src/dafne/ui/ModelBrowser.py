import sys
import webbrowser
from copy import copy

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QDialog, QTreeWidgetItem, QApplication, QTableWidgetItem

from .ModelBrowserUI import Ui_ModelBrowser

class ModelBrowser(QDialog, Ui_ModelBrowser):
    def __init__(self, selected_list, details_dict, parent=None, model_for_info=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(lambda: self.exit_dialog(True))
        self.buttonBox.rejected.connect(lambda: self.exit_dialog(False))
        self.details_table.cellDoubleClicked.connect(self.onCellDoubleClicked)
        self.accepted = False
        self.details_dict = details_dict
        self.selected_set = set(selected_list)
        if model_for_info:
            self.buttonBox.button(self.buttonBox.Ok).setText('Close')
            self.buttonBox.button(self.buttonBox.Cancel).hide()
            self.model_tree.hide()
            self.resize(600, 400)
            self.setWindowTitle(f'Model Information: {model_for_info}')
            self.show()
            self.show_info(model_for_info)
            return

        self.populate()
        self.model_tree.itemClicked.connect(self.item_clicked)
        self.model_tree.itemChanged.connect(self.item_changed)
        self.show()

    def populate(self):
        for model_name, model_details in self.details_dict.items():
            if 'categories' not in model_details:
                model_details['categories'] = [['Unknown']]

            for category in model_details['categories']:
                # Add or find the root item
                group_item = self.find_or_create_root_item(category[0])
                for subgroup in category[1:]:
                    # Add or find the sub-group
                    group_item = self.find_or_create_child_item(group_item, subgroup)

                # Add the final item
                self.find_or_create_child_item(group_item, model_name, is_checkable=True, is_checked=model_name in self.selected_set)

            self.set_bold_when_child_checked()

    def find_or_create_root_item(self, name):
        # Search for the item
        for i in range(self.model_tree.topLevelItemCount()):
            item = self.model_tree.topLevelItem(i)
            if item.text(0) == name:
                return item

        # If not found, create a new root item
        new_item = QTreeWidgetItem(self.model_tree)
        new_item.setText(0, name)
        return new_item

    def set_bold_when_child_checked(self):
        def set_bold(item, is_bold):
            font = item.font(0)
            font.setBold(is_bold)
            item.setFont(0, font)

        def set_bold_when_child_checked_recursive(item):
            if item.childCount() == 0:
                return item.checkState(0) == Qt.Checked

            # Check if any children are checked
            any_checked = False
            for i in range(item.childCount()):
                child = item.child(i)
                if set_bold_when_child_checked_recursive(child):
                    any_checked = True
                    break

            # Set the font to bold if any children are checked
            set_bold(item, any_checked)
            return any_checked

        for i in range(self.model_tree.topLevelItemCount()):
            item = self.model_tree.topLevelItem(i)
            any_checked = set_bold_when_child_checked_recursive(item)
            set_bold(item, any_checked)

    def find_or_create_child_item(self, parent, name, is_checkable=False, is_checked=False):
        # Search for the item
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.text(0) == name:
                return child

        # If not found, create a new child item
        new_child = QTreeWidgetItem(parent)
        new_child.setText(0, name)

        if is_checkable:
            new_child.setFlags(new_child.flags() | Qt.ItemIsUserCheckable)
            if is_checked:
                new_child.setCheckState(0, Qt.Checked)
            else:
                new_child.setCheckState(0, Qt.Unchecked)

        return new_child

    def onCellDoubleClicked(self, row, column):
        # Check if it's the "Value" column
        if column == 1:
            item = self.details_table.item(row, column)
            value = item.text()
            if value.startswith('http://') or value.startswith('https://'):
                webbrowser.open(value)

    def add_table_row(self, name, value):
        row = self.details_table.rowCount()
        self.details_table.insertRow(row)
        value_item = QTableWidgetItem(value)
        value_item.setTextAlignment(Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap)
        value_item.setToolTip(value)
        if value.startswith('http://') or value.startswith('https://'):
            font = value_item.font()
            font.setUnderline(True)
            value_item.setFont(font)
            value_item.setForeground(QColor("blue"))
            value_item.setToolTip("Double-click to open link")


        self.details_table.setItem(row, 0, QTableWidgetItem(name + '   '))
        self.details_table.setItem(row, 1, value_item)


    def show_info(self, model_name):
        def add_to_string(string, value):
            if string != '':
                string += ', '
            string += value
            return string

        self.details_table.setRowCount(0)
        self.add_table_row('Name', model_name)
        try:
            info = self.details_dict[model_name]['info']
        except KeyError:
            pass
        else:
            for name, value in self.details_dict[model_name]['info'].items():
                self.add_table_row(name, value)

        try:
            variants = self.details_dict[model_name]['variants']
        except KeyError:
            pass
        else:
            variant_string = ''
            for variant in variants:
                if variant == '':
                    variant_string = add_to_string(variant_string, 'Default')
                else:
                    variant_string = add_to_string(variant_string, variant)
            self.add_table_row('Variants', variant_string)

        if self.details_table.rowCount() == 1:
            self.add_table_row('', 'No information available')

        self.details_table.resizeColumnToContents(0)
        self.details_table.resizeRowsToContents()

    def item_clicked(self, item, column):
        self.details_table.setRowCount(0)
        if item.childCount() > 0:
            # don't display details for non-leaf items
            return
        model_name = item.text(0)
        self.show_info(model_name)


    def item_changed(self, item, column):
        #change all items with the same name
        model_name = item.text(0)
        status = item.checkState(0) == Qt.Checked
        if status:
            self.selected_set.add(model_name)
        else:
            self.selected_set.discard(model_name)

        def set_checked_recursive(item, text, status):
            if item.childCount() == 0:
                if item.text(0) == text:
                    item.setCheckState(0, Qt.Checked if status else Qt.Unchecked)
                return

            for i in range(item.childCount()):
                child = item.child(i)
                set_checked_recursive(child, text, status)

        for i in range(self.model_tree.topLevelItemCount()):
            item = self.model_tree.topLevelItem(i)
            set_checked_recursive(item, model_name, status)

        self.set_bold_when_child_checked()

    def exit_dialog(self, accepted):
        self.accepted = accepted
        self.close()

    def closeEvent(self, event):
        # set the output
        pass


def show_model_browser(selected_list, details_dict, parent=None):
    dialog = ModelBrowser(selected_list, details_dict, parent)
    dialog.exec_()
    return dialog.accepted, sorted(list(dialog.selected_set))


def show_model_info(model_name, details_dict, parent=None):
    dialog = ModelBrowser([], details_dict, parent, model_name)
    dialog.exec_()

def test():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    model_details = {
        'Leg': {
            'categories': [['MSK', 'Muscle', 'Lower limbs'], ['MRI', 'Axial']],
            'info': {
                'Description': 'Leg model',
                'Author': 'Dafne team',
                'Modality': 'MRI',
                'Orientation': 'Axial'
            },
            'variants': ['', 'Left', 'Right']
        },
        'Kidney': {
            'categories': [['Abdomen'], ['MRI', 'Coronal']],
            'info': {
                'Description': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor.',
                'Author': 'Sheffield team',
                'Modality': 'MRI',
                'Orientation': 'Coronal',
                'Link': 'https://www.google.com/'
            }
        },
        'Brain': { }
    }
    selected_list = ['Leg']
    print(show_model_browser(selected_list, model_details))
    print(show_model_info('Kidney', model_details))