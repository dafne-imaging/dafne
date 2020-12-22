from PyQt5.QtCore import QRunnable, pyqtSlot, QThreadPool
from functools import wraps

threadpool = QThreadPool()

class Runner(QRunnable):

    def __init__(self, func, *args, **kwargs):
        QRunnable.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.func(*self.args, **self.kwargs)

def separate_thread_decorator(func):
    @wraps(func)
    def run_wrapper(*args, **kwargs):
        runner = Runner(func, *args, **kwargs)
        threadpool.start(runner)
    return run_wrapper