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
import time
import logging
import pprint

from IBMQuantumExperience import ApiError
from qiskit._compiler import compile_circuit
from qiskit.backends import BaseJob
from qiskit.backends.basejob import JobStatus
from qiskit._qiskiterror import QISKitError
from qiskit._result import Result
from qiskit._resulterror import ResultError

logger = logging.getLogger(__name__)


class IBMQJob(BaseJob):
    """IBM Q Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, q_job, api, is_device):
        """IBMQJob init function.

        Args:
            q_job (QuantumJob): job description
            api (IBMQuantumExperience): IBM Q API
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj
        """
        super().__init__()
        self._q_job = q_job
        self._qobj = q_job.qobj
        self._api = api
        self._job_id = None  # this must be before creating the future
        self._backend_name = self._qobj.get('config').get('backend_name')
        self._status = JobStatus.INITIALIZING
        self._future_submit = self._executor.submit(self._submit)
        self._status_msg = None
        self._cancelled = False
        self._exception = None
        self._is_device = is_device

    def result(self, timeout=None, wait=5):
        # pylint: disable=arguments-differ
        while self._status == JobStatus.INITIALIZING:
            time.sleep(0.1)
        this_result = self._wait_for_job(timeout=timeout, wait=wait)
        if self._is_device and self.done:
            _reorder_bits(this_result)  # TODO: remove this after Qobj
        if this_result.get_status() == 'ERROR':
            self._status = JobStatus.ERROR
        else:
            self._status = JobStatus.DONE
        return this_result

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
            self._cancelled = False
            return False

    @property
    def status(self):
        if self._status == JobStatus.INITIALIZING:
            stats = {'job_id': None,
                     'status': self._status,
                     'status_msg': 'job is begin initialized please wait a moment'}
            return stats
        job_result = self._api.get_job(self._job_id)
        stats = {'job_id': self._job_id}
        self._status = None
        _status_msg = None
        if 'status' not in job_result:
            self._exception = QISKitError("get_job didn't return status: %s" %
                                          (pprint.pformat(job_result)))
            raise QISKitError("get_job didn't return status: %s" %
                              (pprint.pformat(job_result)))
        elif job_result['status'] == 'RUNNING':
            self._status = JobStatus.RUNNING
            # we may have some other information here
            if 'infoQueue' in job_result:
                if 'status' in job_result['infoQueue']:
                    if job_result['infoQueue']['status'] == 'PENDING_IN_QUEUE':
                        self._status = JobStatus.QUEUED
                if 'position' in job_result['infoQueue']:
                    stats['queue_position'] = job_result['infoQueue']['position']
        elif job_result['status'] == 'COMPLETED':
            self._status = JobStatus.DONE
        elif self.cancelled:
            self._status = JobStatus.CANCELLED
        elif self.exception:
            self._status = JobStatus.ERROR
            if self._future_submit.exception():
                self._exception = self._future_submit.exception()
            self._status_msg = str(self.exception)
        elif 'ERROR' in job_result['status']:
            # ERROR_CREATING_JOB or ERROR_RUNNING_JOB
            self._status = JobStatus.ERROR
            self._status_msg = job_result['status']
        else:
            self._status = JobStatus.ERROR
            raise IBMQJobError('Unexpected behavior of {0}\n{1}'.format(
                self.__class__.__name__,
                pprint.pformat(job_result)))
        stats['status'] = self._status
        stats['status_msg'] = _status_msg
        return stats

    @property
    def queued(self):
        """
        Returns whether job is queued.

        Returns:
            bool: True if job is queued, else False.

        Raises:
            QISKitError: couldn't get job status from server
        """
        return self.status['status'] == JobStatus.QUEUED

    @property
    def running(self):
        """
        Returns whether job is actively running

        Returns:
            bool: True if job is running, else False.

        Raises:
            QISKitError: couldn't get job status from server
        """
        return self.status['status'] == JobStatus.RUNNING

    @property
    def done(self):
        """
        Returns True if job successfully finished running.

        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.
        """
        return self.status['status'] == JobStatus.DONE

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
        if isinstance(self._exception, Exception):
            self._status_msg = str(self._exception)
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

    @property
    def backend_name(self):
        """
        Return backend name used for this job
        """
        return self._backend_name

    def _submit(self):
        """Submit job to IBM Q.

        Returns:
            dict: submission info including job id from server

        Raises:
            QISKitError: The backend name in the job doesn't match this backend.
            ResultError: If the API reported an error with the submitted job.
            RegisterSizeError: If the requested register size exceeded device
                capability.
        """
        qobj = self._qobj
        api_jobs = []
        for circuit in qobj['circuits']:
            if (('compiled_circuit_qasm' not in circuit) or
                    (circuit['compiled_circuit_qasm'] is None)):
                compiled_circuit = compile_circuit(circuit['circuit'])
                circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)
            if isinstance(circuit['compiled_circuit_qasm'], bytes):
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm'].decode()})
            else:
                api_jobs.append({'qasm': circuit['compiled_circuit_qasm']})

        seed0 = qobj['circuits'][0]['config']['seed']
        hpc = None
        if 'hpc' in qobj['config']:
            try:
                # Use CamelCase when passing the hpc parameters to the API.
                hpc = {
                    'multiShotOptimization':
                        qobj['config']['hpc']['multi_shot_optimization'],
                    'ompNumThreads':
                        qobj['config']['hpc']['omp_num_threads']
                }
            except (KeyError, TypeError):
                hpc = None

        backend_name = qobj['config']['backend_name']
        if backend_name != self._backend_name:
            raise QISKitError("inconsistent qobj backend "
                              "name ({0} != {1})".format(backend_name,
                                                         self._backend_name))
        submit_info = {}
        try:
            submit_info = self._api.run_job(api_jobs, backend_name,
                                            shots=qobj['config']['shots'],
                                            max_credits=qobj['config']['max_credits'],
                                            seed=seed0,
                                            hpc=hpc)
        except ApiError as err:
            self._status = JobStatus.ERROR
            self._exception = err
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._exception = IBMQJobError(str(submit_info['error']))
        self._job_id = submit_info.get('id')
        self._status = JobStatus.QUEUED
        return submit_info

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
        api_result = self._api.get_job(job_id)
        while not (self.done or self.cancelled or self.exception):
            if timeout is not None and timer >= timeout:
                job_result = {'job_id': job_id, 'status': 'ERROR',
                              'result': 'QISkit Time Out'}
                return Result(job_result, qobj)
            time.sleep(wait)
            timer += wait
            logger.info('status = %s (%d seconds)', api_result['status'], timer)
            api_result = self._api.get_job(job_id)

            if 'status' not in api_result:
                self._exception = QISKitError("get_job didn't return status: %s" %
                                              (pprint.pformat(api_result)))
                raise QISKitError("get_job didn't return status: %s" %
                                  (pprint.pformat(api_result)))
            if (api_result['status'] == 'ERROR_CREATING_JOB' or
                    api_result['status'] == 'ERROR_RUNNING_JOB'):
                job_result = {'job_id': job_id, 'status': 'ERROR',
                              'result': api_result['status']}
                return Result(job_result, qobj)

        if self.cancelled:
            job_result = {'job_id': job_id, 'status': 'CANCELLED',
                          'result': 'job cancelled'}
            return Result(job_result, qobj)
        elif self.exception:
            job_result = {'job_id': job_id, 'status': 'ERROR',
                          'result': str(self.exception)}
            return Result(job_result, qobj)
        api_result = self._api.get_job(job_id)
        job_result_return = []
        for index in range(len(api_result['qasms'])):
            job_result_return.append({'data': api_result['qasms'][index]['data'],
                                      'status': api_result['qasms'][index]['status']})
        job_result = {'job_id': job_id, 'status': api_result['status'],
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


def _reorder_bits(result):
    """temporary fix for ibmq backends.
    for every ran circuit, get reordering information from qobj
    and apply reordering on result"""
    for idx, circ in enumerate(result._qobj['circuits']):

        # device_qubit -> device_clbit (how it should have been)
        measure_dict = {op['qubits'][0]: op['clbits'][0]
                        for op in circ['compiled_circuit']['operations']
                        if op['name'] == 'measure'}

        res = result._result['result'][idx]
        counts_dict_new = {}
        for item in res['data']['counts'].items():
            # fix clbit ordering to what it should have been
            bits = list(item[0])
            bits.reverse()  # lsb in 0th position
            count = item[1]
            reordered_bits = list('x' * len(bits))
            for device_clbit, bit in enumerate(bits):
                if device_clbit in measure_dict:
                    correct_device_clbit = measure_dict[device_clbit]
                    reordered_bits[correct_device_clbit] = bit
            reordered_bits.reverse()

            # only keep the clbits specified by circuit, not everything on device
            num_clbits = circ['compiled_circuit']['header']['number_of_clbits']
            compact_key = reordered_bits[-num_clbits:]
            compact_key = "".join([b if b != 'x' else '0'
                                   for b in compact_key])

            # insert spaces to signify different classical registers
            cregs = circ['compiled_circuit']['header']['clbit_labels']
            if sum([creg[1] for creg in cregs]) != num_clbits:
                raise ResultError("creg sizes don't add up in result header.")
            creg_begin_pos = []
            creg_end_pos = []
            acc = 0
            for creg in reversed(cregs):
                creg_size = creg[1]
                creg_begin_pos.append(acc)
                creg_end_pos.append(acc + creg_size)
                acc += creg_size
            compact_key = " ".join([compact_key[creg_begin_pos[i]:creg_end_pos[i]]
                                    for i in range(len(cregs))])

            # marginalize over unwanted measured qubits
            if compact_key not in counts_dict_new:
                counts_dict_new[compact_key] = count
            else:
                counts_dict_new[compact_key] += count

        res['data']['counts'] = counts_dict_new
