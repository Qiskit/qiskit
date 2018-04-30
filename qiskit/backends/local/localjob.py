# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""This module implements the job class used for LocalBackend objects."""

from concurrent import futures
import logging

from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit import QISKitError

logger = logging.getLogger(__name__)


class LocalJob(BaseJob):
    """Local QISKit SDK Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, qobj):
        super().__init__()
        self._qobj = qobj
        self._future = self._executor.submit(fn, qobj)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """
        Get job result.

        Returns:
            Result: Result object
        """
        return self._future.result(timeout=timeout)

    def cancel(self):
        return self._future.cancel()

    @property
    def status(self):
        _status_msg = None
        # order is important here
        if self.running:
            _status = JobStatus.RUNNING
        elif not self.done:
            _status = JobStatus.QUEUED
        elif self.cancelled:
            _status = JobStatus.CANCELLED
        elif self.done:
            _status = JobStatus.DONE
        elif isinstance(self.error, Exception):
            _status = JobStatus.ERROR
            _status_msg = str(self.error)
        else:
            raise LocalJobError('Unexpected behavior of {0}'.format(
                self.__class__.__name__))
        _status_msg = None
        return {'status': _status,
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

    @property
    def error(self):
        """
        Return Exception object if exception occured else None.

        Returns:
            Exception: exception raised by attempting to run job.
        """
        return self._future.exception(timeout=0)


class LocalJobError(QISKitError):
    """class for Local Job errors"""
    pass
