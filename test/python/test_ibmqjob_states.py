# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,missing-docstring,broad-except


"""IBMQJob states test-suite."""

import unittest
import time
from IBMQuantumExperience import ApiError
from qiskit.backends.jobstatus import JobStatus
from qiskit.backends.ibmq.ibmqjob import IBMQJob
from qiskit.backends.ibmq.ibmqjob import API_FINAL_STATES
from qiskit.backends import JobError, JobTimeoutError
from .common import QiskitTestCase
from ._mockutils import new_fake_qobj


class TestIBMQJobStates(QiskitTestCase):
    """
    Test ibmqjob module.
    """
    def setUp(self):
        self._current_api = None
        self._current_qjob = None

    def test_unrecognized_status(self):
        job = self.run_with_api(UnknownStatusAPI())
        with self.assertRaises(JobError):
            self.wait_for_initialization(job)

    def test_validating_job(self):
        job = self.run_with_api(ValidatingAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.VALIDATING)

    def test_error_while_creating_job(self):
        job = self.run_with_api(ErrorWhileCreatingAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_error_while_running_job(self):
        job = self.run_with_api(ErrorWhileRunningAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.ERROR)

        # We have an error message this time
        self.assertEqual(job.error_message(), 'Error running job')

    def test_error_while_validating_job(self):
        job = self.run_with_api(ErrorWhileValidatingAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.VALIDATING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_status_flow_for_non_queued_job(self):
        job = self.run_with_api(NonQueuedAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_status_flow_for_queued_job(self):
        job = self.run_with_api(QueuedAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.QUEUED)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_status_flow_for_cancellable_job(self):
        job = self.run_with_api(CancellableAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertTrue(can_cancel)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.CANCELLED)

    def test_status_flow_for_non_cancellable_job(self):
        job = self.run_with_api(NonCancellableAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertFalse(can_cancel)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.RUNNING)

    def test_status_flow_for_errored_cancellation(self):
        """This is still on RUNNING state because the Job
        couldn't be cancelled, and is not due to a server
        internal error, it's just in a non-cancelable state,
        so the server returns an ERROR response.
        """
        job = self.run_with_api(ErroredCancellationAPI())
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)
        can_cancel = job.cancel()
        self.assertFalse(can_cancel)
        self.assertEqual(job.status(), JobStatus.RUNNING)

    def test_status_flow_for_invalid_job(self):
        job = self.run_with_api(UnableToInitializeAPI())
        # TODO This is very risky, if the status changes to ERROR too fast
        # this assert will fail, but it's not a bug in the code.
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_status_flow_for_throwing_job(self):
        """ If there's an exception in the thread that will submit the job, it
        doesn't mean the Job had a problem, it might be a temporary server specific
        error. So the job status will remain unchanged, but an exception will be
        re-thrown when calling status().
        If something wrong actually happened to the Job, is the server responsibility
        to return the ERROR state, but the status() method will not raise an
        exception in this case.
        """
        job = self.run_with_api(ThrowingServerButJobFinishedAPI())
        # We need to give time to the thread to populate the exception, before we
        # can re-throwing it in status()
        time.sleep(0.2)
        with self.assertRaises(JobError):
            job.status()

    def test_status_flow_for_throwing_api(self):
        job = self.run_with_api(ThrowingAPI())
        with self.assertRaises(JobError):
            self.wait_for_initialization(job)

    def test_cancelled_result(self):
        job = self.run_with_api(CancellableAPI())

        self.wait_for_initialization(job)
        job.cancel()
        self._current_api.progress()
        self.assertEqual(job.result().get_status(), 'CANCELLED')
        self.assertEqual(job.status(), JobStatus.CANCELLED)

    def test_errored_result(self):
        job = self.run_with_api(ThrowingGetJobAPI())
        self.wait_for_initialization(job)
        with self.assertRaises(JobError):
            job.result()

    def test_completed_result(self):
        job = self.run_with_api(NonQueuedAPI())

        self.wait_for_initialization(job)
        self._current_api.progress()
        self.assertEqual(job.result().get_status(), 'COMPLETED')
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_block_on_result_waiting_until_completed(self):
        from concurrent import futures

        job = self.run_with_api(NonQueuedAPI())
        with futures.ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_block_on_result_waiting_until_cancelled(self):
        from concurrent.futures import ThreadPoolExecutor

        job = self.run_with_api(CancellableAPI())
        with ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.get_status(), 'CANCELLED')
        self.assertEqual(job.status(), JobStatus.CANCELLED)

    def test_block_on_result_waiting_until_exception(self):
        from concurrent.futures import ThreadPoolExecutor
        job = self.run_with_api(ThrowingAPI())

        with ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        with self.assertRaises(JobError):
            job.result()

    def test_never_complete_result_with_timeout(self):
        job = self.run_with_api(NonQueuedAPI())

        self.wait_for_initialization(job)
        # We never make the API status to progress so it is stuck on RUNNING
        with self.assertRaises(JobTimeoutError):
            job.result(timeout=0.2)

        self.assertEqual(job.status(), JobStatus.RUNNING)

    def test_cancel_while_initializing_fails(self):
        job = self.run_with_api(CancellableAPI())
        can_cancel = job.cancel()
        self.assertFalse(can_cancel)
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

    def test_only_final_states_cause_datailed_request(self):
        """The state ERROR_CREATING_JOB is only handled when running the job,
        and not while checking the status, so it is not tested.
        """
        from unittest import mock
        all_state_apis = {'COMPLETED': NonQueuedAPI,
                          'CANCELLED': CancellableAPI,
                          'ERROR_VALIDATING_JOB': ErrorWhileValidatingAPI,
                          'ERROR_RUNNING_JOB': ErrorWhileRunningAPI}

        for status, api in all_state_apis.items():
            with self.subTest(status=status):
                job = self.run_with_api(api())
                self.wait_for_initialization(job)

                try:
                    self._current_api.progress()
                except BaseFakeAPI.NoMoreStatesError:
                    pass

                with mock.patch.object(self._current_api, 'get_job',
                                       wraps=self._current_api.get_job):
                    job.status()
                    if status in API_FINAL_STATES:
                        self.assertTrue(self._current_api.get_job.called)
                    else:
                        self.assertFalse(self._current_api.get_job.called)

    def test_job_has_finished_but_api_throws_temporarily(self):
        """There could be temporary problems with the API server, but doesn't
        mean the job has failed. The server could start working fine again, and
        we should be able to retrieve the job"""

        job = self.run_with_api(TemporaryThrowingServerButJobFinishedAPI())
        job._wait_for_submission()

        # First time we query the server, we get an exception...
        with self.assertRaises(JobError):
            job.status()

        # Maybe, also a second time...
        with self.assertRaises(JobError):
            job.status()

        # Now the API server gets restored and doesn't throw anymore
        # The job has finished successfully
        self.assertEqual(job.status(), JobStatus.DONE)

    def wait_for_initialization(self, job, timeout=1):
        """Waits until the job progress from `INITIALIZING` to a different
        status.
        """
        waited = 0
        wait = 0.1
        while job.status() == JobStatus.INITIALIZING:
            time.sleep(wait)
            waited += wait
            if waited > timeout:
                self.fail(
                    msg="The JOB is still initializing after timeout ({}s)"
                    .format(timeout)
                )

    def run_with_api(self, api):
        """Creates a new `IBMQJob` instance running with the provided API
        object.
        """
        self._current_api = api
        self._current_qjob = IBMQJob(api, False, qobj=new_fake_qobj())
        self._current_qjob.submit()
        return self._current_qjob


def _auto_progress_api(api, interval=0.2):
    """Progress a `BaseFakeAPI` instacn every `interval` seconds until reaching
    the final state.
    """
    try:
        while True:
            time.sleep(interval)
            api.progress()
    except BaseFakeAPI.NoMoreStatesError:
        pass


class BaseFakeAPI():
    """Base class for faking the IBM-Q API."""

    class NoMoreStatesError(Exception):
        """Raised when it is not possible to progress more."""

    _job_status = []

    _can_cancel = False

    def __init__(self):
        self._state = 0
        self.config = {'hub': None, 'group': None, 'project': None}
        if self._can_cancel:
            self.config.update({
                'hub': 'test-hub',
                'group': 'test-group',
                'project': 'test-project'
            })

    def get_job(self, job_id):
        if not job_id:
            return {'status': 'Error', 'error': 'Job ID not specified'}
        return self._job_status[self._state]

    def get_status_job(self, job_id):
        summary_fields = ['status', 'error', 'infoQueue']
        complete_response = self.get_job(job_id)
        return {key: value for key, value in complete_response.items()
                if key in summary_fields}

    def run_job(self, *_args, **_kwargs):
        time.sleep(0.2)
        return {'id': 'TEST_ID'}

    def cancel_job(self, job_id, *_args, **_kwargs):
        if not job_id:
            return {'status': 'Error', 'error': 'Job ID not specified'}
        return {} if self._can_cancel else {
            'error': 'testing fake API can not cancel'}

    def progress(self):
        if self._state == len(self._job_status) - 1:
            raise self.NoMoreStatesError()
        self._state += 1


class UnknownStatusAPI(BaseFakeAPI):
    """Class for emulating an API with unknown status codes."""

    _job_status = [
        {'status': 'UNKNOWN'}
    ]


class ValidatingAPI(BaseFakeAPI):
    """Class for emulating an API with job validation."""

    _job_status = [
        {'status': 'VALIDATING'},
        {'status': 'RUNNING'}
    ]


class ErrorWhileValidatingAPI(BaseFakeAPI):
    """Class for emulating an API processing an invalid job."""

    _job_status = [
        {'status': 'VALIDATING'},
        {'status': 'ERROR_VALIDATING_JOB'}
    ]


class NonQueuedAPI(BaseFakeAPI):
    """Class for emulating a successfully-completed non-queued API."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'COMPLETED', 'qasms': []}
    ]


class ErrorWhileCreatingAPI(BaseFakeAPI):
    """Class emulating an API processing a job that errors while creating
    the job.
    """

    _job_status = [
        {'status': 'ERROR_CREATING_JOB'}
    ]


class ErrorWhileRunningAPI(BaseFakeAPI):
    """Class emulating an API processing a job that errors while running."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'ERROR_RUNNING_JOB', 'error': 'Error running job'}
    ]


class QueuedAPI(BaseFakeAPI):
    """Class for emulating a successfully-completed queued API."""

    _job_status = [
        {'status': 'RUNNING', 'infoQueue': {'status': 'PENDING_IN_QUEUE'}},
        {'status': 'RUNNING'},
        {'status': 'COMPLETED'}
    ]


class UnableToInitializeAPI(BaseFakeAPI):
    """Class for emulating an API unable of initializing."""

    def run_job(self, *_args, **_kwargs):
        time.sleep(0.2)
        return {'error': 'invalid test qobj'}


class ThrowingInitializationAPI(BaseFakeAPI):
    """Class for emulating an API throwing before even initializing."""

    def run_job(self, *_args, **_kwargs):
        time.sleep(0.2)
        raise ApiError()


class ThrowingAPI(BaseFakeAPI):
    """Class for emulating an API throwing in the middle of execution."""

    _job_status = [
        {'status': 'RUNNING'}
    ]

    def get_job(self, job_id):
        raise ApiError()


class ThrowingServerButJobFinishedAPI(BaseFakeAPI):
    """Class for emulating an API throwing in the middle of execution, but
    corresponding Job completes. This can simulate a server exception, but
    not a a job problem, indeed, this will simulate a successfully completed
    job.
    """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    def get_status_job(self, job_id):
        return self._job_status[self._state]

    def run_job(self, *_args, **_kwargs):
        raise ApiError()


class TemporaryThrowingServerButJobFinishedAPI(BaseFakeAPI):
    """Class for emulating an API throwing only a few times, but the job successfully
    finishes. We pretend to simulate a temporary server malfunction that gets eventually
    fixed, so in the end, we get the job.
    """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    _number_of_exceptions_to_throw = 2

    def get_job(self, job_id):
        if self._number_of_exceptions_to_throw != 0:
            self._number_of_exceptions_to_throw -= 1
            raise ApiError()

        return super().get_job(job_id)


class ThrowingGetJobAPI(BaseFakeAPI):
    """Class for emulating an API throwing in the middle of execution. But not in
       get_status_job() , just in get_job().
       """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    def get_status_job(self, job_id):
        return self._job_status[self._state]

    def get_job(self, job_id):
        raise ApiError("Unexpected error")


class CancellableAPI(BaseFakeAPI):
    """Class for emulating an API with cancellation."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'CANCELLED'}
    ]

    _can_cancel = True


class NonCancellableAPI(BaseFakeAPI):
    """Class for emulating an API without cancellation running a long job."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'RUNNING'},
        {'status': 'RUNNING'}
    ]


class ErroredCancellationAPI(BaseFakeAPI):
    """Class for emulating an API with cancellation but throwing while
    trying.
    """

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'RUNNING'},
        {'status': 'RUNNING'}
    ]

    _can_cancel = True

    def cancel_job(self, job_id, *_args, **_kwargs):
        return {'status': 'Error', 'error': 'test-error-while-cancelling'}


if __name__ == '__main__':
    unittest.main(verbosity=2)
