# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""IBMQJob module

This module is used for creating asynchronous job objects for the
IBM Q Experience.
"""

from concurrent import futures
import time
import logging
import pprint
import json
import datetime
import numpy

from IBMQuantumExperience import ApiError

from qiskit.qobj import qobj_to_dict
from qiskit.transpiler import transpile
from qiskit.backends import BaseJob, JobError
from qiskit.backends.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit._result import Result
from qiskit._resulterror import ResultError

logger = logging.getLogger(__name__)


API_FINAL_STATES = (
    'COMPLETED',
    'CANCELLED',
    'ERROR_CREATING_JOB',
    'ERROR_VALIDATING_JOB',
    'ERROR_RUNNING_JOB'
)


class IBMQJob(BaseJob):
    """IBM Q Job class

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
        _final_states (list(JobStatus)): terminal states of async jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, api, is_device, qobj=None, job_id=None, backend_name=None,
                 creation_date=None):
        """IBMQJob init function.
        We can instantiate Jobs from two sources: A QObj, and an already submitted job returned by
        the API servers.

        Args:
            api (IBMQuantumExperience): IBM Q API
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj
            qobj (Qobj): The Quantum Object. If not present, is because we are probably getting
                         a Job from the API server.
            job_id (String): The Job ID of an already submitted job.
            backend_name(String): The name of the backend that run the job.
            creation_date(String): When the Job was run.
        """
        super().__init__()
        # TODO No need for this conversion, just use the new equivalent members above
        if qobj is not None:
            old_qobj = qobj_to_dict(qobj, version='0.0.1')
            self._job_data = {
                'circuits': old_qobj['circuits'],
                'hpc':  None if 'hpc' not in old_qobj['config'] else old_qobj['config']['hpc'],
                'seed': old_qobj['circuits'][0]['config']['seed'],  # TODO <-- [0] ???
                'shots': old_qobj['config']['shots'],
                'max_credits': old_qobj['config']['max_credits']
            }
        self._api = api
        self._id = job_id  # this must be before creating the future
        self._backend_name = qobj.header.backend_name if qobj is not None else backend_name
        # If we are creating the Job from a QObj, then it's ok to initialize the status to
        # INITIALIZING, otherwise we are creating a Job from an already submitted job, so
        # we have to ask the API server in which state the job is found
        self._status = JobStatus.INITIALIZING
        if qobj is None:
            self.status()
        self._queue_position = 0
        self._cancelled = False
        self._is_device = is_device
        self._creation_date = datetime.datetime.utcnow().replace(
            tzinfo=datetime.timezone.utc).isoformat() if creation_date is None else creation_date
        self._future = None

    # pylint: disable=arguments-differ
    def result(self, timeout=None, wait=5):
        """Return the result from the job.

        Args:
           timeout (int): number of seconds to wait for job
           wait (int): time between queries to IBM Q server

        Returns:
            Result: Result object

        Raises:
            JobError: exception raised during job initialization
        """
        self._wait_for_submitting()
        try:
            this_result = self._wait_for_job(timeout=timeout, wait=wait)
        except TimeoutError as err:
            # A timeout error retrieving the results does not imply the job
            # is failing. The job can be still running.
            return Result({'id': self._id, 'status': 'ERROR',
                           'result': str(err)})

        if self._is_device and self.done:
            _reorder_bits(this_result)

        return this_result

    def cancel(self):
        """Attempt to cancel a job. If it is not possible, check the
        ```exception``` property for more information.

        Returns:
            bool: True if job can be cancelled, else False. Currently this is
            only possible on commercial systems.

        Raises:
            JobError: if server returned error
        """
        hub = self._api.config.get('hub', None)
        group = self._api.config.get('group', None)
        project = self._api.config.get('project', None)

        cancelled = False
        try:
            response = self._api.cancel_job(self._id, hub, group, project)
            errored = 'error' in response
            cancelled = not errored

            if errored:
                self._cancelled = cancelled
                err_msg = response.get('error', '')
                raise JobError('Error cancelling job: %s' % err_msg)

        except ApiError as error:
            self._cancelled = cancelled
            err_msg = error.usr_msg
            raise JobError('Error cancelling job: %s' % err_msg)

        return self._cancelled

    def status(self):
        """Query the API to update the status.

        Returns:
            JobStatus: The status of the job, once updated.

        Raises:
            JobError: if there was an exception in the future being executed
                          or the server sent an unknown answer.
        """

        if self._status in JOB_FINAL_STATES or self._id is None:
            return self._status

        try:
            # TODO See result values
            api_job = self._api.get_status_job(self._id)
            if api_job['status'] in API_FINAL_STATES:
                # Call the endpoint that returns full information.
                api_job = self._api.get_job(self._id)

            if 'status' not in api_job:
                self._status = JobStatus.ERROR
                raise JobError('get_job didn\'t return status: %s' %
                               pprint.pformat(api_job))
        # pylint: disable=broad-except
        except Exception as err:
            self._status = JobStatus.ERROR
            raise JobError(str(err))

        if api_job['status'] == 'VALIDATING':
            self._status = JobStatus.VALIDATING

        elif api_job['status'] == 'RUNNING':
            self._status = JobStatus.RUNNING
            queued, self._queue_position = _is_job_queued(api_job)
            if queued:
                self._status = JobStatus.QUEUED

        elif api_job['status'] == 'COMPLETED':
            self._status = JobStatus.DONE

        elif api_job['status'] == 'CANCELLED':
            self._status = JobStatus.CANCELLED
            self._cancelled = True

        elif 'ERROR' in api_job['status']:
            # Error statuses are of the form "ERROR_*_JOB"
            self._status = JobStatus.ERROR

        else:
            self._status = JobStatus.ERROR
            raise JobError('Unrecognized answer from server: \n{}'
                           .format(pprint.pformat(api_job)))

        return self._status

    def queue_position(self):
        """
        Returns the position in the server queue

        Returns:
            Number: Position in the queue. 0 = No queued.
        """
        return self._queue_position

    @property
    def queued(self):
        """
        Returns whether job is queued.

        Returns:
            bool: True if job is queued, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.QUEUED

    @property
    def running(self):
        """
        Returns whether job is actively running

        Returns:
            bool: True if job is running, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.RUNNING

    @property
    def validating(self):
        """
        Returns whether job is being validated

        Returns:
            bool: True if job is under validation, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.VALIDATING

    @property
    def done(self):
        """
        Returns True if job successfully finished running.
        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.

        Return:
            bool: True if job successfully finished running.

        Raises:
            IMBQJobError: couldn't get job status from server
        """
        return self.status() == JobStatus.DONE

    @property
    def cancelled(self):
        return self._cancelled

    # pylint: disable=invalid-name
    @property
    def id(self):
        """
        Return backend determined id.
        If the ID is not set because the job is already initializing, this call will
        block until we have an ID.
        """
        self._wait_for_submitting()
        return self._id

    def _is_commercial(self):
        """ Returns True if the job is running in one of the commercial backends """
        config = self._api.config
        # this check may give false positives so should probably be improved
        return config.get('hub') and config.get('group') and config.get('project')

    def backend_name(self):
        """
        Return backend name used for this job
        """
        return self._backend_name

    def submit(self):
        """Submit job to IBM Q.

        Returns:
            dict: submission info including job id from server

        Raises:
            ResultError: If the API reported an error with the submitted job.
            RegisterSizeError: If the requested register size exceeded device
                capability.
            JobError: If we have already submited the job.
        """
        if self._future is not None or self._id is not None:
            raise JobError("We have already submitted the job!")

        api_jobs = []
        circuits = self._job_data['circuits']
        for circuit in circuits:
            job = _create_job_from_circuit(circuit)
            api_jobs.append(job)

        hpc = self._job_data['hpc']
        seed = self._job_data['seed']
        shots = self._job_data['shots']
        max_credits = self._job_data['max_credits']

        hpc_camel_cased = None
        if hpc is not None:
            try:
                # Use CamelCase when passing the hpc parameters to the API.
                hpc_camel_cased = {
                    'multiShotOptimization': hpc['multi_shot_optimization'],
                    'ompNumThreads': hpc['omp_num_threads']
                }
            except (KeyError, TypeError):
                hpc_camel_cased = None

        self._future = self._executor.submit(self._submit_callback, api_jobs,
                                             self._backend_name, hpc_camel_cased,
                                             seed, shots, max_credits)

    def _submit_callback(self, api_jobs, backend_name, hpc, seed, shots, max_credits):
        """Submit job to IBM Q.

        Args:
            api_jobs (list): List of API Job dictionaries to submit. One per circuit.
            backend_name (string): The name of the backend
            hpc (dict): HPC specific configuration
            seed (integer): The seed for the circuits
            shots (integer): Number of shots the circuits should run
            max_credits (integer): Maximum number of credits

        Return:
            A dictionary with the response of the submitted job

        Raises:
            JobError: If something bad happened during job request to the API
            ResultError: If the API reported an error with the submitted job.
            RegisterSizeError: If the requested register size exceeded device
                capability.
        """
        try:
            submit_info = self._api.run_job(api_jobs, backend=backend_name,
                                            shots=shots, max_credits=max_credits,
                                            seed=seed, hpc=hpc)
        except Exception as err:
            self._status = JobStatus.ERROR
            raise JobError(str(err))

        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            return submit_info

        self._id = submit_info.get('id')
        self._creation_date = submit_info.get('creationDate')
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
            TimeoutError: if the job does not return results before an
            specified timeout.
        """
        start_time = time.time()
        while self.status() not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise TimeoutError('QISKit timed out')

            logger.info('status = %s (%d seconds)', self._status, elapsed_time)
            time.sleep(wait)

        if self.cancelled:
            return Result({'id': self._id, 'status': 'CANCELLED',
                           'result': 'job cancelled'})

        job_data = self._api.get_job(self._id)
        job_result_list = []
        for circuit_result in job_data['qasms']:
            this_result = {'data': circuit_result['data'],
                           'name': circuit_result.get('name'),
                           'compiled_circuit_qasm': circuit_result.get('qasm'),
                           'status': circuit_result['status']}
            if 'metadata' in circuit_result:
                this_result['metadata'] = circuit_result['metadata']

            job_result_list.append(this_result)

        return Result({'id': self._id,
                       'status': job_data['status'],
                       'used_credits': job_data.get('usedCredits'),
                       'result': job_result_list,
                       'backend_name': self.backend_name})

    def _wait_for_submitting(self):
        """ Waits for the request to return a Job ID """
        if self._id is None:
            if self._future is None:
                raise JobError("You have to submit before asking for status or results!")
            submit_info = self._future.result(timeout=60)
            if 'error' in submit_info:
                self._status = JobStatus.ERROR
                raise JobError(str(submit_info['error']))

            self._creation_date = submit_info.get('creationDate')
            self._status = JobStatus.QUEUED
            self._id = submit_info.get('id')


def _reorder_bits(result):
    """temporary fix for ibmq backends.
    for every ran circuit, get reordering information from qobj
    and apply reordering on result"""
    for circuit_result in result._result['result']:
        if 'metadata' in circuit_result:
            circ = circuit_result['metadata'].get('compiled_circuit')
        else:
            logger.warning('result object missing metadata for reordering'
                           ' bits: bits may be out of order')
            return
        # device_qubit -> device_clbit (how it should have been)
        measure_dict = {op['qubits'][0]: op['clbits'][0]
                        for op in circ['operations']
                        if op['name'] == 'measure'}
        counts_dict_new = {}
        for item in circuit_result['data']['counts'].items():
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
            num_clbits = circ['header']['number_of_clbits']
            compact_key = reordered_bits[-num_clbits:]
            compact_key = "".join([b if b != 'x' else '0'
                                   for b in compact_key])

            # insert spaces to signify different classical registers
            cregs = circ['header']['clbit_labels']
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

        circuit_result['data']['counts'] = counts_dict_new


def _numpy_type_converter(obj):
    if isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):  # pylint: disable=no-member
        return float(obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    return obj


def _create_job_from_circuit(circuit):
    """ Helper function that creates a special Job required by the API, from a circuit """
    job = {}
    if not circuit.get('compiled_circuit_qasm', None):
        compiled_circuit = transpile(circuit['circuit'])
        circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)

    if isinstance(circuit['compiled_circuit_qasm'], bytes):
        job['qasm'] = circuit['compiled_circuit_qasm'].decode()
    else:
        job['qasm'] = circuit['compiled_circuit_qasm']

    if circuit.get('name', None):
        job['name'] = circuit['name']

    # convert numpy types for json serialization
    compiled_circuit = json.loads(json.dumps(circuit['compiled_circuit'],
                                             default=_numpy_type_converter))

    job['metadata'] = {'compiled_circuit': compiled_circuit}
    return job


def _is_job_queued(api_job):
    """ Checks whether a Job has been queued or not """
    is_queued, position = False, 0
    if 'infoQueue' in api_job:
        if 'status' in api_job['infoQueue']:
            queue_status = api_job['infoQueue']['status']
            is_queued = queue_status == 'PENDING_IN_QUEUE'
        if 'position' in api_job['infoQueue']:
            position = api_job['infoQueue']['position']
    return is_queued, position
