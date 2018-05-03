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
"""IBMQJob module

This module is used for creating asynchronous job objects for the
IBM Q Experience.
"""

from concurrent import futures
import logging

from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit._qiskiterror import QISKitError
from qiskit._result import Result

logger = logging.getLogger(__name__)


class IBMQJob(BaseJob):
    """IBM Q Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, fn, q_job, api, timeout, submit_info):
        """IBMQJob init function.

        Args:
            fn (function): function object to call for job
            q_job (QuantumJob): job description
            api (IBMQuantumExperience): IBM Q API
            timeout (float): timeout in seconds
            submit_info (dict): this is the dictionary that comes back when
                submitting a job using the IBMQuantumExperience API.
        """
        super().__init__()
        self._q_job = q_job
        self._api = api
        self._timeout = timeout
        self._submit_info = submit_info
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._status_msg = submit_info['error']['message']
            self._job_id = None
        else:
            self._job_id = submit_info['id']
            self._future = self._executor.submit(fn, q_job)
            self._status = JobStatus.QUEUED
            self._status_msg = None

    def result(self, timeout=None):
        # pylint: disable=arguments-differ
        try:
            return self._future.result(timeout=timeout)
        except futures.TimeoutError as err:
            qobj = self._q_job.qobj
            qobj_result = {'backend_name': qobj['backend_name'],
                           'header': qobj['config'],
                           'job_id': self._job_id,
                           'results': []}
            for circ in qobj['circuits']:
                seed = circ['config'].get('seed',
                                          qobj['config'].get('seed', None))
                shots = circ['config'].get('shots',
                                           qobj['config'].get('shots', None))
                exp_result = {'shots': shots,
                              'status': str(err),
                              'success': False,
                              'seed': seed,
                              'data': None}
                qobj_result['results'].append(exp_result)
            return Result(qobj_result, qobj)

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
        # check for submission error
        if self._status == JobStatus.ERROR:
            return {'job_id': self._job_id,
                    'status': self._status,
                    'status_msg': self._status_msg}
        else:
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
                raise IBMQJobError('Unexpected behavior of {0}'.format(
                    self.__class__.__name__))
            return {'job_id': self._job_id,
                    'status': _status,
                    'status_msg': _status_msg}

    @property
    def running(self):
        """
        Returns whether job is actively running

        Returns:
            bool: True if job is running, else False.
        """
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
    def error(self):
        """
        Return Exception object if exception occured else None.

        Returns:
            Exception: exception raised by attempting to run job.
        """
        return self._future.exception(timeout=0)

    @property
    def _is_commercial(self):
        config = self._api.config
        # this check may give false positives so should probably be improved
        return config['hub'] and config['group'] and config['project']

    @property
    def job_id(self):
        """
        Return backend determined job_id (also available in status method).
        """
        return self._job_id


class IBMQJobError(QISKitError):
    """class for IBM Q Job errors"""
    pass
