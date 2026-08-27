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

from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QThreadPool, QObject, QMutex, QWaitCondition, QThread, Qt
from functools import wraps
import traceback

threadpool = QThreadPool()


class MainThreadDialogRunner(QObject):
    """ Runs a callable (typically one that creates/executes a Qt dialog) on the
        thread that constructed this object (the GUI thread, if instantiated at
        module import time as the singleton below), blocking the calling thread
        until it completes, and returning its result.

        Qt widgets may only be created/shown on the GUI thread; code running in a
        @separate_thread_decorator-wrapped method must go through this (rather
        than instantiating a dialog directly) to stay thread-safe. """

    _run_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._result = None
        self._run_signal.connect(self._run, Qt.QueuedConnection)

    @pyqtSlot(object)
    def _run(self, factory):
        try:
            self._result = factory()
        except Exception:
            traceback.print_exc()
            self._result = None
        finally:
            self._mutex.lock()
            self._condition.wakeAll()
            self._mutex.unlock()

    def run(self, factory):
        if QThread.currentThread() is self.thread():
            return factory()
        self._mutex.lock()
        self._run_signal.emit(factory)
        self._condition.wait(self._mutex)
        result = self._result
        self._mutex.unlock()
        return result


# constructed at import time (main thread), so its thread affinity is the GUI thread
main_thread_dialog_runner = MainThreadDialogRunner()

class Runner(QRunnable):

    def __init__(self, func, *args, **kwargs):
        QRunnable.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            setattr(self.args[0], 'separate_thread_running', True)
        except:
            pass
        self.func(*self.args, **self.kwargs)
        try:
            setattr(self.args[0], 'separate_thread_running', False)
        except:
            pass

def separate_thread_decorator(func):
    @wraps(func)
    def run_wrapper(*args, **kwargs):
        runner = Runner(func, *args, **kwargs)
        threadpool.start(runner)
    return run_wrapper