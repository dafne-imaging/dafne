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

from PyQt5.QtCore import QRunnable, pyqtSlot, QThreadPool
from functools import wraps
import traceback

threadpool = QThreadPool()

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