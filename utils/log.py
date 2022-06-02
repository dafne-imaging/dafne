from PyQt5.QtCore import QObject, pyqtSignal


class LogStream(QObject):

    updated = pyqtSignal(str)

    def __init__(self, file, old_descriptor = None, parent=None):
        super(LogStream, self).__init__(parent)
        self.fdesc = open(file, 'w')
        self.old_descriptor = old_descriptor
        self.data = ''

    def write(self, data):
        self.fdesc.write(data)
        self.fdesc.flush()
        if self.old_descriptor is not None:
            self.old_descriptor.write(data)
            self.old_descriptor.flush()
        self.data += data
        self.updated.emit(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, item):
        return getattr(self.fdesc, item)

    def get_data(self):
        return self.data

    def close(self):
        self.fdesc.close()

log_objects = {}
