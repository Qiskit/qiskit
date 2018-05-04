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

import time
import logging
import pprint

from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit._qiskiterror import QISKitError
from qiskit._result import Result

logger = logging.getLogger(__name__)


class IBMQJob(BaseJob):
    """IBM Q Job class"""

    def __init__(self, q_job, api, submit_info):
        """IBMQJob init function.

        Args:
            q_job (QuantumJob): job description
            api (IBMQuantumExperience): IBM Q API
            submit_info (dict): this is the dictionary that comes back when
                submitting a job using the IBMQuantumExperience API.
        """
        super().__init__()
        self._q_job = q_job
        self._api = api
        self._submit_info = submit_info
        self._job_id = submit_info['id']
        self._status = JobStatus.QUEUED
        self._status_msg = None
        self._cancelled = False
        self._exception = None

    def result(self, timeout=None, wait=5):
        # pylint: disable=arguments-differ
        return self._wait_for_job(timeout=timeout, wait=wait)

    def cancel(self):
        """Attempt to cancel job. Currently this is only possible on
        commercial systems.

        Returns:
            bool: True if job can be cancelled, else False.

        Raises:
            QISKitError: if server returned error
        """
        if self._is_commercial:
            hub = self._api.config['hub']
            group = self._api.config['group']
            project = self._api.config['project']
            response = self._api.cancel_job(self._job_id, hub, group, project)
            if 'error' in response:
                err_msg = response.get('error', '')
                self._exception = QISKitError('Error cancelling job: %s' % err_msg)
                raise QISKitError('Error canceelling job: %s' % err_msg)
            else:
                self._cancelled = True
                return True
        else:
            # Since currently the API always reports RUNNING it isn't possible
            # for the SDK to know when the job is actually running instead of just
            # sitting in queue. Once that changes it may be useful to reimplement
            # this class.
            return False

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
            elif self.exception:
                _status = JobStatus.ERROR
                _status_msg = str(self.exception)
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

        Raises:
            QISKitError: couldn't get job status from server
        """
        job_result = self._api.get_job(self._job_id)
        if 'status' not in job_result:
            self._exception = QISKitError("get_job didn't return status: %s" %
                                          (pprint.pformat(job_result)))
            raise QISKitError("get_job didn't return status: %s" %
                              (pprint.pformat(job_result)))
        return job_result['status'] == 'RUNNING'

    @property
    def done(self):
        """
        Returns True if job successfully finished running.

        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.
        """
        job_result = self._api.get_job(self._job_id)
        if 'status' not in job_result:
            self._exception = QISKitError("get_job didn't return status: %s" %
                                          (pprint.pformat(job_result)))
            raise QISKitError("get_job didn't return status: %s" %
                              (pprint.pformat(job_result)))
        return job_result['status'] == 'COMPLETED'

    @property
    def cancelled(self):
        return self._cancelled

    @property
    def exception(self):
        """
        Return Exception object previously raised by job else None

        Returns:
            Exception: exception raised by job
        """
        return self._exception

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

    def _wait_for_job(self, timeout=60, wait=5):
        """Wait until all online ran circuits of a qobj are 'COMPLETED'.

        Args:
            timeout (float or None): seconds to wait for job. If None, wait
                indefinitely.
            wait (float): seconds between queries

        Returns:
            Result: A result object.

        Raises:
            QISKitError: job didn't return status or reported error in status
        """
        qobj = self._q_job.qobj
        job_id = self.job_id
        logger.info('Running qobj: %s on remote backend %s with job id: %s',
                    qobj["id"], qobj['config']['backend_name'],
                    job_id)
        timer = 0
        job_result = self._api.get_job(job_id)
        while self.running:
            if timeout is not None and timer >= timeout:
                job_result = {'job_id': job_id, 'status': 'ERROR',
                              'result': 'QISkit Time Out'}
                return Result(job_result, qobj)
            time.sleep(wait)
            timer += wait
            logger.info('status = %s (%d seconds)', job_result['status'], timer)
            job_result = self._api.get_job(job_id)

            if 'status' not in job_result:
                self._exception = QISKitError("get_job didn't return status: %s" %
                                              (pprint.pformat(job_result)))
                raise QISKitError("get_job didn't return status: %s" %
                                  (pprint.pformat(job_result)))
            if (job_result['status'] == 'ERROR_CREATING_JOB' or
                    job_result['status'] == 'ERROR_RUNNING_JOB'):
                job_result = {'job_id': job_id, 'status': 'ERROR',
                              'result': job_result['status']}
                return Result(job_result, qobj)

        # Get the results
        job_result_return = []
        for index in range(len(job_result['qasms'])):
            job_result_return.append({'data': job_result['qasms'][index]['data'],
                                      'status': job_result['qasms'][index]['status']})
        job_result = {'job_id': job_id, 'status': job_result['status'],
                      'result': job_result_return}
        logger.info('Got a result for qobj: %s from remote backend %s with job id: %s',
                    qobj["id"], qobj['config']['backend_name'],
                    job_id)
        job_result['name'] = qobj['id']
        job_result['backend'] = qobj['config']['backend_name']
        return Result(job_result, qobj)


class IBMQJobError(QISKitError):
    """class for IBM Q Job errors"""
    pass
