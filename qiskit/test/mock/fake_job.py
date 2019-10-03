# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Base class for dummy jobs.
"""


from concurrent import futures

from qiskit.providers import BaseJob
from qiskit.providers.jobstatus import JobStatus


class FakeJob(BaseJob):
    """Fake simulator job"""
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, backend, job_id, fn, qobj):
        super().__init__(backend, job_id)
        self._backend = backend
        self._job_id = job_id
        self._qobj = qobj
        self._future = None
        self._future_callback = fn

    def submit(self):
        self._future = self._executor.submit(self._future_callback, self._qobj)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    def status(self):
        if self._running:
            _status = JobStatus.RUNNING
        elif not self._done:
            _status = JobStatus.QUEUED
        elif self._cancelled:
            _status = JobStatus.CANCELLED
        elif self._done:
            _status = JobStatus.DONE
        elif self._error:
            _status = JobStatus.ERROR
        else:
            raise Exception('Unexpected state of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
                'status_msg': _status_msg}

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

    @property
    def _cancelled(self):
        return self._future.cancelled()

    @property
    def _done(self):
        return self._future.done()

    @property
    def _running(self):
        return self._future.running()

    @property
    def _error(self):
        return self._future.exception(timeout=0)
