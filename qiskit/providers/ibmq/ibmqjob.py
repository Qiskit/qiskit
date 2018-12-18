# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""IBMQJob module

This module is used for creating asynchronous job objects for the
IBM Q Experience.
"""

import warnings
from concurrent import futures
import time
import logging
import pprint
import contextlib
import json
import datetime
import numpy

from qiskit.qobj import qobj_to_dict, Qobj
from qiskit.transpiler import transpile_dag
from qiskit.providers import BaseJob, JobError, JobTimeoutError
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.result._utils import result_from_old_style_dict
from qiskit.qobj import validate_qobj_against_schema

from .api import ApiError


logger = logging.getLogger(__name__)


API_FINAL_STATES = (
    'COMPLETED',
    'CANCELLED',
    'ERROR_CREATING_JOB',
    'ERROR_VALIDATING_JOB',
    'ERROR_RUNNING_JOB'
)


class IBMQJob(BaseJob):
    """Represent the jobs that will be executed on IBM-Q simulators and real
    devices. Jobs are intended to be created calling ``run()`` on a particular
    backend.

    Creating a ``Job`` instance does not imply running it. You need to do it in
    separate steps::

        job = IBMQJob(...)
        job.submit() # It won't block.

    An error while submitting a job will cause the next call to ``status()`` to
    raise. If submitting the job successes, you can inspect the job's status by
    using ``status()``. Status can be one of ``JobStatus`` members::

        from qiskit.backends.jobstatus import JobStatus

        job = IBMQJob(...)
        job.submit()

        try:
            job_status = job.status() # It won't block. It will query the backend API.
            if job_status is JobStatus.RUNNING:
                print('The job is still running')

        except JobError as ex:
            print("Something wrong happened!: {}".format(ex))

    A call to ``status()`` can raise if something happens at the API level that
    prevents Qiskit from determining the status of the job. An example of this
    is a temporary connection lose or a network failure.

    The ``submit()`` and ``status()`` methods are examples of non-blocking API.
    ``Job`` instances also have `id()` and ``result()`` methods which will
    block::

        job = IBMQJob(...)
        job.submit()

        try:
            job_id = job.id() # It will block until completing submission.
            print('The job {} was successfully submitted'.format(job_id))

            job_result = job.result() # It will block until finishing.
            print('The job finished with result {}'.format(job_result))

        except JobError as ex:
            print("Something wrong happened!: {}".format(ex))

    Both methods can raise if something ath the API level happens that prevent
    Qiskit from determining the status of the job.

    Note:
        When querying the API for getting the status, two kinds of errors are
        possible. The most severe is the one preventing Qiskit from getting a
        response from the backend. This can be caused by a network failure or a
        temporary system break. In these cases, calling ``status()`` will raise.

        If Qiskit successfully retrieves the status of a job, it could be it
        finished with errors. In that case, ``status()`` will simply return
        ``JobStatus.ERROR`` and you can call ``error_message()`` to get more
        info.

    Attributes:
        _executor (futures.Executor): executor to handle asynchronous jobs
    """
    _executor = futures.ThreadPoolExecutor()

    def __init__(self, backend, job_id, api, is_device, qobj=None,
                 creation_date=None, api_status=None):
        """IBMQJob init function.

        We can instantiate jobs from two sources: A QObj, and an already submitted job returned by
        the API servers.

        Args:
            backend (str): The backend instance used to run this job.
            job_id (str): The job ID of an already submitted job. Pass `None`
                if you are creating a new one.
            api (IBMQConnector): IBMQ connector.
            is_device (bool): whether backend is a real device  # TODO: remove this after Qobj
            qobj (Qobj): The Quantum Object. See notes below
            creation_date (str): When the job was run.
            api_status (str): `status` field directly from the API response.

        Notes:
            It is mandatory to pass either ``qobj`` or ``job_id``. Passing a ``qobj``
            will ignore ``job_id`` and will create an instance to be submitted to the
            API server for job creation. Passing only a `job_id` will create an instance
            representing an already-created job retrieved from the API server.
        """
        super().__init__(backend, job_id)
        self._job_data = None

        if qobj is not None:
            validate_qobj_against_schema(qobj)

            self._qobj_payload = qobj_to_dict(qobj, version='1.0.0')
            # TODO: No need for this conversion, just use the new equivalent members above
            old_qobj = qobj_to_dict(qobj, version='0.0.1')
            self._job_data = {
                'circuits': old_qobj['circuits'],
                'hpc':  old_qobj['config'].get('hpc'),
                'seed': old_qobj['circuits'][0]['config']['seed'],
                'shots': old_qobj['config']['shots'],
                'max_credits': old_qobj['config']['max_credits']
            }
        else:
            self._qobj_payload = {}

        self._future_captured_exception = None
        self._api = api
        self._backend = backend
        self._cancelled = False
        self._status = JobStatus.INITIALIZING
        # In case of not providing a `qobj`, it is assumed the job already
        # exists in the API (with `job_id`).
        if qobj is None:
            # Some API calls (`get_status_jobs`, `get_status_job`) provide
            # enough information to recreate the `Job`. If that is the case, try
            # to make use of that information during instantiation, as
            # `self.status()` involves an extra call to the API.
            if api_status == 'VALIDATING':
                self._status = JobStatus.VALIDATING
            elif api_status == 'COMPLETED':
                self._status = JobStatus.DONE
            elif api_status == 'CANCELLED':
                self._status = JobStatus.CANCELLED
                self._cancelled = True
            else:
                self.status()
        self._queue_position = None
        self._is_device = is_device

        def current_utc_time():
            """Gets the current time in UTC format"""
            datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

        self._creation_date = creation_date or current_utc_time()
        self._future = None
        self._api_error_msg = None

    # pylint: disable=arguments-differ
    def result(self, timeout=None, wait=5):
        """Return the result from the job.

        Args:
           timeout (int): number of seconds to wait for job
           wait (int): time between queries to IBM Q server

        Returns:
            qiskit.Result: Result object

        Raises:
            JobError: exception raised during job initialization
        """
        job_response = self._wait_for_result(timeout=timeout, wait=wait)
        return self._result_from_job_response(job_response)

    def _wait_for_result(self, timeout=None, wait=5):
        self._wait_for_submission(timeout)

        try:
            job_response = self._wait_for_job(timeout=timeout, wait=wait)
            if not self._qobj_payload:
                self._qobj_payload = job_response.get('qObject', {})
        except ApiError as api_err:
            raise JobError(str(api_err))

        status = self.status()
        if status is not JobStatus.DONE:
            raise JobError('Invalid job state. The job should be DONE but '
                           'it is {}'.format(str(status)))

        return job_response

    def _result_from_job_response(self, job_response):
        return Result.from_dict(job_response['qObjectResult'])

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
            response = self._api.cancel_job(self._job_id, hub, group, project)
            self._cancelled = 'error' not in response
            return self._cancelled
        except ApiError as error:
            self._cancelled = False
            raise JobError('Error cancelling job: %s' % error.usr_msg)

    def status(self):
        """Query the API to update the status.

        Returns:
            qiskit.providers.JobStatus: The status of the job, once updated.

        Raises:
            JobError: if there was an exception in the future being executed
                          or the server sent an unknown answer.
        """
        # Implies self._job_id is None
        if self._future_captured_exception is not None:
            raise JobError(str(self._future_captured_exception))

        if self._job_id is None or self._status in JOB_FINAL_STATES:
            return self._status

        try:
            # TODO: See result values
            api_job = self._api.get_status_job(self._job_id)
            if 'status' not in api_job:
                raise JobError('get_status_job didn\'t return status: %s' %
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
            # TODO: This seems to be an inconsistency in the API package.
            self._api_error_msg = api_job.get('error') or api_job.get('Error')

        else:
            raise JobError('Unrecognized answer from server: \n{}'
                           .format(pprint.pformat(api_job)))

        return self._status

    def error_message(self):
        """Return the error message returned from the API server response."""
        return self._api_error_msg

    def queue_position(self):
        """Return the position in the server queue.

        Returns:
            Number: Position in the queue.
        """
        return self._queue_position

    def creation_date(self):
        """
        Return creation date.
        """
        return self._creation_date

    def job_id(self):
        """Return backend determined id.

        If the Id is not set because the job is already initializing, this call
        will block until we have an Id.
        """
        self._wait_for_submission()
        return self._job_id

    def submit(self):
        """Submit job to IBM-Q.

        Raises:
            JobError: If we have already submitted the job.
        """
        # TODO: Validation against the schema should be done here and not
        # during initialization. Once done, we should document that the method
        # can raise QobjValidationError.
        if self._future is not None or self._job_id is not None:
            raise JobError("We have already submitted the job!")
        self._future = self._executor.submit(self._submit_callback)

    def _submit_callback(self):
        """Submit qobj job to IBM-Q.

        Returns:
            dict: A dictionary with the response of the submitted job
        """
        backend_name = self.backend().name()

        try:
            submit_info = self._api.run_job(self._qobj_payload, backend=backend_name)
        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            self._future_captured_exception = err
            return None

        # Error in the job after submission:
        # Transition to the `ERROR` final state.
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._api_error_msg = str(submit_info['error'])
            return submit_info

        # Submission success.
        self._creation_date = submit_info.get('creationDate')
        self._status = JobStatus.QUEUED
        self._job_id = submit_info.get('id')
        return submit_info

    def _wait_for_job(self, timeout=60, wait=5):
        """Wait until all online ran circuits of a qobj are 'COMPLETED'.

        Args:
            timeout (float or None): seconds to wait for job. If None, wait
                indefinitely.
            wait (float): seconds between queries

        Returns:
            dict: A dict with the contents of the API request.

        Raises:
            JobTimeoutError: if the job does not return results before a specified timeout.
            JobError: if something wrong happened in some of the server API calls
        """
        start_time = time.time()
        while self.status() not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    'Timeout while waiting for the job: {}'.format(self._job_id)
                )

            logger.info('status = %s (%d seconds)', self._status, elapsed_time)
            time.sleep(wait)

        if self._cancelled:
            raise JobError(
                'Job result impossible to retrieve. The job was cancelled.')

        return self._api.get_job(self._job_id)

    def _wait_for_submission(self, timeout=60):
        """Waits for the request to return a job ID"""
        if self._job_id is None:
            if self._future is None:
                raise JobError("You have to submit before asking for status or results!")
            try:
                submit_info = self._future.result(timeout=timeout)
                if self._future_captured_exception is not None:
                    # pylint can't see if catch of None type
                    # pylint: disable=raising-bad-type
                    raise self._future_captured_exception
            except TimeoutError as ex:
                raise JobTimeoutError(
                    "Timeout waiting for the job being submitted: {}".format(ex)
                )
            if 'error' in submit_info:
                self._status = JobStatus.ERROR
                self._api_error_msg = str(submit_info['error'])
                raise JobError(str(submit_info['error']))

    def qobj(self):
        """Return the Qobj submitted for this job.

        Note that this method might involve querying the API for results if the
        Job has been created in a previous Qiskit session.

        Returns:
            Qobj: the Qobj submitted for this job.
        """
        if not self._qobj_payload:
            # Populate self._qobj_payload by retrieving the results.
            self._wait_for_result()

        return Qobj(**self._qobj_payload)


class IBMQJobPreQobj(IBMQJob):
    """Subclass of IBMQJob for handling pre-qobj jobs."""

    def _submit_callback(self):
        """Submit old style qasms job to IBM-Q. Can remove when all devices
        understand Qobj.

        Returns:
            dict: A dictionary with the response of the submitted job
        """
        api_jobs = []
        circuits = self._job_data['circuits']
        for circuit in circuits:
            job = _create_api_job_from_circuit(circuit)
            api_jobs.append(job)

        hpc_camel_cased = _format_hpc_parameters(self._job_data['hpc'])
        seed = self._job_data['seed']
        shots = self._job_data['shots']
        max_credits = self._job_data['max_credits']

        try:
            submit_info = self._api.run_job(api_jobs, backend=self.backend().name(),
                                            shots=shots, max_credits=max_credits,
                                            seed=seed, hpc=hpc_camel_cased)
        # pylint: disable=broad-except
        except Exception as err:
            # Undefined error during submission:
            # Capture and keep it for raising it when calling status().
            self._future_captured_exception = err
            return None

        # Error in the job after submission:
        # Transition to the `ERROR` final state.
        if 'error' in submit_info:
            self._status = JobStatus.ERROR
            self._api_error_msg = str(submit_info['error'])
            return submit_info

        # Submission success.
        self._creation_date = submit_info.get('creationDate')
        self._status = JobStatus.QUEUED
        self._job_id = submit_info.get('id')
        return submit_info

    def _result_from_job_response(self, job_response):
        if self._is_device:
            _reorder_bits(job_response)

        experiment_results = []
        for circuit_result in job_response['qasms']:
            this_result = {'data': circuit_result['data'],
                           'compiled_circuit_qasm': circuit_result.get('qasm'),
                           'status': circuit_result['status'],
                           'success': circuit_result['status'] == 'DONE',
                           'shots': job_response['shots']}
            if 'metadata' in circuit_result:
                this_result['metadata'] = circuit_result['metadata']
                if 'header' in circuit_result['metadata'].get('compiled_circuit', {}):
                    this_result['header'] = \
                        circuit_result['metadata']['compiled_circuit']['header']
                else:
                    this_result['header'] = {}
            experiment_results.append(this_result)

        ret = {
            'id': self._job_id,
            'status': job_response['status'],
            'used_credits': job_response.get('usedCredits'),
            'result': experiment_results,
            'backend_name': self.backend().name(),
            'success': job_response['status'] == 'COMPLETED',
        }

        # Append header: from the response; from the payload; or none.
        header = job_response.get('header',
                                  self._qobj_payload.get('header', {}))
        if header:
            ret['header'] = header

        return result_from_old_style_dict(ret)

    def qobj(self):
        """Return the Qobj submitted for this job."""
        warnings.warn('This job has not been submitted using Qobj.')
        return None


def _reorder_bits(job_data):
    """Temporary fix for ibmq backends.

    For every ran circuit, get reordering information from qobj
    and apply reordering on result.

    Args:
        job_data (dict): dict with the bare contents of the API.get_job request.

    Raises:
        JobError: raised if the creg sizes don't add up in result header.
    """
    for circuit_result in job_data['qasms']:
        if 'metadata' in circuit_result:
            circ = circuit_result['metadata'].get('compiled_circuit')
        else:
            logger.warning('result object missing metadata for reordering'
                           ' bits: bits may be out of order')
            return
        # device_qubit -> device_clbit (how it should have been)
        measure_dict = {op['qubits'][0]: op.get('clbits', op.get('memory'))[0]
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
            num_clbits = circ['header'].get('number_of_clbits',
                                            circ['header'].get('memory_slots'))
            compact_key = reordered_bits[-num_clbits:]
            compact_key = "".join([b if b != 'x' else '0'
                                   for b in compact_key])

            # insert spaces to signify different classical registers
            cregs = circ['header'].get('creg_sizes',
                                       circ['header'].get('clbit_labels'))
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


def _create_api_job_from_circuit(circuit):
    """Helper function that creates a special job required by the API, from a circuit."""
    api_job = {}
    if not circuit.get('compiled_circuit_qasm'):
        compiled_circuit = transpile_dag(circuit['circuit'])
        circuit['compiled_circuit_qasm'] = compiled_circuit.qasm(qeflag=True)

    if isinstance(circuit['compiled_circuit_qasm'], bytes):
        api_job['qasm'] = circuit['compiled_circuit_qasm'].decode()
    else:
        api_job['qasm'] = circuit['compiled_circuit_qasm']

    if circuit.get('name'):
        api_job['name'] = circuit['name']

    # convert numpy types for json serialization
    compiled_circuit = json.loads(json.dumps(circuit['compiled_circuit'],
                                             default=_numpy_type_converter))

    api_job['metadata'] = {'compiled_circuit': compiled_circuit}
    return api_job


def _is_job_queued(api_job_response):
    """Checks whether a job has been queued or not."""
    is_queued, position = False, 0
    if 'infoQueue' in api_job_response:
        if 'status' in api_job_response['infoQueue']:
            queue_status = api_job_response['infoQueue']['status']
            is_queued = queue_status == 'PENDING_IN_QUEUE'
        if 'position' in api_job_response['infoQueue']:
            position = api_job_response['infoQueue']['position']
    return is_queued, position


def _format_hpc_parameters(hpc):
    """Helper function to get HPC parameters with the correct format"""
    if hpc is None:
        return None

    hpc_camel_cased = None
    with contextlib.suppress(KeyError, TypeError):
        # Use CamelCase when passing the hpc parameters to the API.
        hpc_camel_cased = {
            'multiShotOptimization': hpc['multi_shot_optimization'],
            'ompNumThreads': hpc['omp_num_threads']
        }

    return hpc_camel_cased
