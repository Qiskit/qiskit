from concurrent import futures
import logging

from qiskit.backends import BaseJob

logger = logging.getLogger(__name__)

class LocalJob(BaseJob):
    """Local QISKit SDK Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ProcessPoolExecutor()
    
    def __init__(self, fn, qobj):
        self._qobj = qobj
        self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def cancelled(self):
        return self._future.cancelled()

    def done(self):
        return self._future.done()
    
    def status(self):
        if self.running():
            return "running"
        elif self.cancelled():
            return "cancelled"
        elif self.done():
            return "done"
        else:
            return "unknown"

    def cancel():
        return self._future.cancel()

    def running(self):
        return self._future.running()

    def add_done_callback(self, fn):
        self._future.add_done_callback(fn)
