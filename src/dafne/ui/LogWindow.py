from PyQt5 import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog

from ..ui.LogWindowUI import Ui_LogWindow

from ..utils.log import log_objects


class LogWindow(QDialog, Ui_LogWindow):
    def __init__(self, parent=None):
        super(LogWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Log")
        self.refresh_btn.clicked.connect(self.refresh)
        self.resize(1024, 768)
        self.refresh()
        try:
            log_objects['stdout'].updated.connect(self.append_output)
        except KeyError:
            pass

        try:
            log_objects['stderr'].updated.connect(self.append_error)
        except KeyError:
            pass

    @pyqtSlot(str)
    def append_output(self, data):
        self.output_text.moveCursor(Qt.QTextCursor.End)
        self.output_text.insertPlainText(data)
        self.output_text.moveCursor(Qt.QTextCursor.End)

    @pyqtSlot(str)
    def append_error(self, data):
        self.error_text.moveCursor(Qt.QTextCursor.End)
        self.error_text.insertPlainText(data)
        self.error_text.moveCursor(Qt.QTextCursor.End)

    def refresh(self):
        self.output_text.clear()
        self.error_text.clear()

        if 'stdout' not in log_objects:
            self.output_text.appendPlainText("Not available")
        else:
            self.output_text.appendPlainText(log_objects['stdout'].get_data())

        if 'stderr' not in log_objects:
            self.error_text.appendPlainText("Not available")
        else:
            self.error_text.appendPlainText(log_objects['stderr'].get_data())

        self.output_text.moveCursor(Qt.QTextCursor.End)
        self.error_text.moveCursor(Qt.QTextCursor.End)


