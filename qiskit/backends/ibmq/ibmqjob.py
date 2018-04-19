from concurrent import futures
import logging

from qiskit.backends import BaseJob
from qiskit._qiskiterror import QISKitError

logger = logging.getLogger(__name__)

class IBMQJob(BaseJob):
    """IBM Q Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, fn, qobj, api, timeout, submit_info):
        self._api = api
        self._timeout = timeout
        self._submit_info = submit_info
        if 'error' in submit_info:
            self._status = 'ERROR'
            self._status_msg = submit_info['error']['message']
        else:
            self._job_id = submit_info['id']
            self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        return self._future.result(timeout=timeout)

    def cancel(self):
        """Attempt to cancel job. Currently this is only possible on 
        commercial systems.
        """
        if self._is_commercial:
            hub = self._api.config['hub']
            group = self._api.config['group']
            project = self._api.config['project']
            response = self._api.cancel_job(self._job_id, hub, group, project)
            if 'error' in response:
                err_msg = response.get('error', '')
                raise QISKitError('Error cancelling job: %s' % err_msg)
        raise QISKitError('The IBM Q remote API does not currently implement'
                          ' job cancellation')

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
            raise IBMQJobError('Unexpected behavior of {0}'.format(
                self.__class__.__name__))
        _status_msg = None # This will be more descriptive
        return {'job_id': self._job_id,
                'status': _status,
                'status_msg': _status_msg} 

    @property
    def running(self):
        return self._future.running()

    @property
    def done(self):
        return self._future.done()

    @property
    def cancelled(self):
        return self._future.cancelled()
        
        
    def _is_commercial(self):
        config = self._api.config
        return config['hub'] and config['group'] and config['project']

class IBMQJobError(QISKitError):
    """class for IBM Q Job errors"""
    pass
