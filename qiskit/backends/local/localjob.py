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
import sys

from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit import QISKitError

logger = logging.getLogger(__name__)


class LocalJob(BaseJob):
    """Local QISKit SDK Job class.

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    if sys.platform == 'darwin':
        _executor = futures.ThreadPoolExecutor()
    else:
        _executor = futures.ProcessPoolExecutor()

    def __init__(self, fn, q_job):
        super().__init__()
        self._q_job = q_job
        self._backend_name = q_job.backend.name
        self._future = self._executor.submit(fn, q_job)

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        """
        Get job result. The behavior is the same as the underlying
        concurrent Future objects,

        https://docs.python.org/3/library/concurrent.futures.html#future-objects

        Args:
            timeout (float): number of seconds to wait for results.

        Returns:
            Result: Result object

        Raises:
            concurrent.futures.TimeoutError: if timeout occured.
            concurrent.futures.CancelledError: if job cancelled before completed.
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
        elif self.exception:
            _status = JobStatus.ERROR
            _status_msg = str(self.exception)
        else:
            raise LocalJobError('Unexpected behavior of {0}'.format(
                self.__class__.__name__))
        return {'status': _status,
                'status_msg': _status_msg}

    @property
    def running(self):
        return self._future.running()

    @property
    def done(self):
        """
        Returns True if job successfully finished running.

        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.
        """
        return self._future.done() and not self._future.cancelled()

    @property
    def cancelled(self):
        return self._future.cancelled()

    @property
    def exception(self):
        """
        Return Exception object if exception occured else None.

        Returns:
            Exception: exception raised by attempting to run job.
        """
        return self._future.exception(timeout=0)

    @property
    def backend_name(self):
        """
        Return backend name used for this job
        """
        return self._backend_name


class LocalJobError(QISKitError):
    """class for Local Job errors"""
    pass
