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
from qiskit.backends import BaseJob, JobError, JobTimeoutError
from qiskit.backends.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit._result import Result

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

    This class represents the jobs that will be executed on IBM Q backends
    (simulators or real chips). It has some asynchronous properties, and some
    blocking behavior. This is a common use-case example:
    .. highlight::

        try:
            job = IBMQJob( ... )
            job.submit() # <--- It won't block.

            while not job.done(): # <---- It will return false if the job
                                  # hasn't finished yet
                pass # <---- or maybe... serialize to disk? .. whatever the user wants

            result = job.result() # <--- This could block, but as the job is already
                                  # finished, it won't in this case.
        except JobError as ex:
            log("Something wrong happened!: {}".format(ex))

    About the status in case of Error or Exception
    ==============================================
    If there's an exception from the IBMQuantumExperience API once a job has been submitted,
    we cannot set the status of the job to ERROR, because it might happen that the job is actually
    queued or running, and let's say that the exception is caused because of a temporary
    out of service situation of the server, once this situation is restored, we might want to ask
    again about the status of our job, that could be, for example, in state: queued, running or
    done. If something wrong happened to the job, it's server responsability to answer an error
    response to a status query from Qiskit. But, if the exception is thrown before having a
    valid job ID, there's nothing we can do with this job instance, as we cannot query the server
    for information using this job instance.

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, api, is_device, qobj=None, job_id=None, backend_name=None,
                 creation_date=None):
        """IBMQJob init function.
        We can instantiate jobs from two sources: A QObj, and an already submitted job returned by
        the API servers.

        Args:
            api (IBMQuantumExperience): IBM Q API
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj
            qobj (Qobj): The Quantum Object. See notes below
            job_id (String): The job ID of an already submitted job.
            backend_name(String): The name of the backend that run the job.
            creation_date(String): When the job was run.

        Notes:
            It is mandatory to pass either ``qobj`` or ``job_id``. Passing a ``qobj``
            will ignore ``job_id`` and will create an instance representing
            an already-created job retrieved from the API server.
        """
        super().__init__()
        self._job_data = None
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
        self._future_exception = None
        self._api = api
        self._id = job_id
        self._backend_name = qobj.header.backend_name if qobj is not None else backend_name
        # If we are creating the job from a QObj, then it's ok to initialize the status to
        # INITIALIZING, otherwise we are creating a job from an already submitted job, so
        # we have to ask the API server in which state the job is found
        self._status = JobStatus.INITIALIZING
        if qobj is None:
            self.status()
        self._queue_position = None
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
        self._wait_for_submission()
        try:
            this_result = self._wait_for_job(timeout=timeout, wait=wait)
        except TimeoutError as err:
            # A timeout error retrieving the results does not imply the job
            # is failing. The job can be still running.
            return Result({'id': self._id, 'status': 'ERROR',
                           'result': str(err)})
        except ApiError as api_err:
            raise JobError(str(api_err))

        if self._is_device and self.done:
            _reorder_bits(this_result)

        return this_result

    def cancel(self):
        """Attempt to cancel a job.

        Returns:
            bool: True if job can be cancelled, else False. Currently this is
            only possible on commercial systems.

        Raises:
            JobError: if there was some unexpected failure in the server
        """
        hub = self._api.config.get('hub', None)
        group = self._api.config.get('group', None)
        project = self._api.config.get('project', None)

        try:
            response = self._api.cancel_job(self._id, hub, group, project)
            self._cancelled = 'error' not in response
            return self._cancelled
        except ApiError as error:
            self._cancelled = False
            raise JobError('Error cancelling job: %s' % error.usr_msg)

    def status(self):
        """Query the API to update the status.

        Returns:
            JobStatus: The status of the job, once updated.

        Raises:
            JobError: if there was an exception in the future being executed
                          or the server sent an unknown answer.
        """

        # This will only happen if something bad happens when submitting a job, so
        # we don't have a job ID
        if self._future_exception is not None:
            raise JobError(str(self._future_exception))

        if self._status in JOB_FINAL_STATES or self._id is None:
            return self._status

        try:
            # TODO See result values
            api_job = self._api.get_status_job(self._id)
            if 'status' not in api_job:
                raise JobError('get_job didn\'t return status: %s' %
                               pprint.pformat(api_job))
        # pylint: disable=broad-except
        except Exception as err:
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
            # Error status are of the form "ERROR_*_JOB"
            self._status = JobStatus.ERROR

        else:
            raise JobError('Unrecognized answer from server: \n{}'
                           .format(pprint.pformat(api_job)))

        return self._status

    def queue_position(self):
        """Returns the position in the server queue

        Returns:
            Number: Position in the queue. 0 = No queued yet
        """
        return self._queue_position

    def creation_date(self):
        """
        Returns creation date
        """
        return self._creation_date

    @property
    def queued(self):
        """Returns whether job is queued.

        Returns:
            bool: True if job is queued, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.QUEUED

    @property
    def running(self):
        """Returns whether job is actively running

        Returns:
            bool: True if job is running, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.RUNNING

    @property
    def validating(self):
        """Returns whether job is being validated

        Returns:
            bool: True if job is under validation, else False.

        Raises:
            JobError: couldn't get job status from server
        """
        return self.status() == JobStatus.VALIDATING

    @property
    def done(self):
        """Returns True if job successfully finished running.

        Note behavior is slightly different than Future objects which would
        also return true if successfully cancelled.

        Returns:
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
        """Return backend determined id.

        If the ID is not set because the job is already initializing, this call will
        block until we have an ID.
        """
        self._wait_for_submission()
        return self._id

    def backend_name(self):
        """Return backend name used for this job"""
        return self._backend_name

    def submit(self):
        """Submit job to IBM Q.

        Raises:
            JobError: If we have already submited the job.
        """
        if self._future is not None and self._id is not None:
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

        hpc_camel_cased = _format_hpc_parameters(hpc)

        self._future = self._executor.submit(self._submit_callback, api_jobs,
                                             self._backend_name, hpc_camel_cased,
                                             seed, shots, max_credits)

    def _submit_callback(self, api_jobs, backend_name, hpc, seed, shots, max_credits):
        """Submit job to IBM Q.

        Args:
            api_jobs (list): List of API job dictionaries to submit. One per circuit.
            backend_name (string): The name of the backend
            hpc (dict): HPC specific configuration
            seed (integer): The seed for the circuits
            shots (integer): Number of shots the circuits should run
            max_credits (integer): Maximum number of credits

        Returns:
            dict: A dictionary with the response of the submitted job
        """
        try:
            submit_info = self._api.run_job(api_jobs, backend=backend_name,
                                            shots=shots, max_credits=max_credits,
                                            seed=seed, hpc=hpc)
        except Exception as err:  # pylint: disable=broad-except
            # This is not a controlled error from the API, so we don't want to change
            # job status to ERROR, because we really don't know what happened with the
            # Job at this point... it could be successfully QUEUED for example.
            self._future_exception = err
            return {'error': str(err)}

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
            JobTimeoutError: if the job does not return results before a specified timeout.
            JobError: if something wrong happened in some of the server API calls
        """
        start_time = time.time()
        while self.status() not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    'Timeout while waiting for the job: {}'.format(self._id)
                )

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

    def _wait_for_submission(self):
        """Waits for the request to return a job ID"""
        if self._id is None:
            if self._future is None:
                raise JobError("You have to submit before asking for status or results!")
            try:
                submit_info = self._future.result(timeout=60)
            except TimeoutError as ex:
                raise JobTimeoutError(
                    "Timeout waiting for the job being submitted: {}".format(ex)
                )

            if 'error' in submit_info:
                self._status = JobStatus.ERROR
                raise JobError(str(submit_info['error']))

            self._creation_date = submit_info.get('creationDate')
            self._status = JobStatus.QUEUED
            self._id = submit_info.get('id')


def _reorder_bits(result):
    """Temporary fix for ibmq backends.

    For every ran circuit, get reordering information from qobj
    and apply reordering on result.
    """
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
                raise JobError("creg sizes don't add up in result header.")
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
    ret = obj
    if isinstance(obj, numpy.integer):
        ret = int(obj)
    elif isinstance(obj, numpy.floating):  # pylint: disable=no-member
        ret = float(obj)
    elif isinstance(obj, numpy.ndarray):
        ret = obj.tolist()
    return ret


def _create_job_from_circuit(circuit):
    """Helper function that creates a special job required by the API, from a circuit."""
    api_job = {}
    if not circuit.get('compiled_circuit_qasm', None):
        compiled_circuit = transpile(circuit['circuit'])
        circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)

    if isinstance(circuit['compiled_circuit_qasm'], bytes):
        api_job['qasm'] = circuit['compiled_circuit_qasm'].decode()
    else:
        api_job['qasm'] = circuit['compiled_circuit_qasm']

    if circuit.get('name', None):
        api_job['name'] = circuit['name']

    # convert numpy types for json serialization
    compiled_circuit = json.loads(json.dumps(circuit['compiled_circuit'],
                                             default=_numpy_type_converter))

    api_job['metadata'] = {'compiled_circuit': compiled_circuit}
    return api_job


def _is_job_queued(api_job):
    """Checks whether a job has been queued or not."""
    is_queued, position = False, 0
    if 'infoQueue' in api_job:
        if 'status' in api_job['infoQueue']:
            queue_status = api_job['infoQueue']['status']
            is_queued = queue_status == 'PENDING_IN_QUEUE'
        if 'position' in api_job['infoQueue']:
            position = api_job['infoQueue']['position']
    return is_queued, position


def _format_hpc_parameters(hpc):
    """Helper function to get HPC parameters with the correct format"""
    if hpc is None:
        return None

    hpc_camel_cased = None
    try:
        # Use CamelCase when passing the hpc parameters to the API.
        hpc_camel_cased = {
            'multiShotOptimization': hpc['multi_shot_optimization'],
            'ompNumThreads': hpc['omp_num_threads']
        }
    except (KeyError, TypeError):
        pass

    return hpc_camel_cased
