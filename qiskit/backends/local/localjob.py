from concurrent import futures
import logging
import uuid

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
        self._job_id = None

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    @property
    def status(self):
        # order is important here
        if self.running:
            _status = 'RUNNING'
        elif not self.done:
            _status = 'QUEUED'
        elif self.cancelled:
            _status = 'CANCELLED'
        elif self.done:
            _status = 'DONE'
        elif self.error:
            _status = 'ERROR'
        else:
            raise LocalJobError('Unexpected behavior of {0}'.format(
                self.__class__.__name__))
        _status_msg = None # This will be more descriptive
        return {'job_id': self._job_id,
                'status': _status,
                'status_msg': _status_msg} 

    @property        
    def running(self):
        return self._future.running()

    @property    
    def cancelled(self):
        return self._future.cancelled()

    @property    
    def done(self):
        return self._future.done()
    
class LocalJobError(QISKitError):
    """class for Local Job errors"""
    pass
    
