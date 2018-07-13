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
from qiskit.backends.ibmq.ibmqjob import IBMQJob, IBMQJobError
from qiskit.backends.ibmq.ibmqjob import API_FINAL_STATES
from qiskit.qobj import Qobj
from .common import QiskitTestCase
from ._mockutils import new_fake_qobj


class TestIBMQJobStates(QiskitTestCase):
    """
    Test ibmqjob module.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def setUp(self):
        self._current_api = None
        self._current_qjob = None

    def test_unrecognized_status(self):
        job = self.run_with_api(UnknownStatusAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)
        self.wait_for_initialization(job)
        self.assertIsInstance(job.exception, IBMQJobError)
        self.assertStatus(job, JobStatus.ERROR)

    def test_validating_job(self):
        job = self.run_with_api(ValidatingAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.VALIDATING)

        self._current_api.progress()

    def test_error_while_creating_job(self):
        job = self.run_with_api(ErrorWhileCreatingAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.ERROR)

    def test_error_while_running_job(self):
        job = self.run_with_api(ErrorWhileRunningAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.RUNNING)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.ERROR)

    def test_error_while_validating_job(self):
        job = self.run_with_api(ErrorWhileValidatingAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.VALIDATING)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.ERROR)

    def test_status_flow_for_non_queued_job(self):
        job = self.run_with_api(NonQueuedAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.RUNNING)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.DONE)

    def test_status_flow_for_queued_job(self):
        job = self.run_with_api(QueuedAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.QUEUED)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.RUNNING)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.DONE)

    def test_status_flow_for_cancellable_job(self):
        job = self.run_with_api(CancellableAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertTrue(can_cancel)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.CANCELLED)

    def test_status_flow_for_non_cancellable_job(self):
        job = self.run_with_api(NonCancellableAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertFalse(can_cancel)

        self._current_api.progress()
        self.assertStatus(job, JobStatus.RUNNING)

    def test_status_flow_for_throwing_cancellation(self):
        job = self.run_with_api(ThrowingCancellationAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertStatus(job, JobStatus.RUNNING)

        with self.assertRaises(IBMQJobError):
            job.cancel()
        self.assertIsInstance(job.exception, IBMQJobError)

        self.assertStatus(job, JobStatus.RUNNING)

    def test_status_flow_for_invalid_job(self):
        job = self.run_with_api(UnableToInitializeAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertIsInstance(job.exception, IBMQJobError)
        self.assertStatus(job, JobStatus.ERROR)

    def test_status_flow_for_throwing_job(self):
        job = self.run_with_api(ThrowingInitializationAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertIsInstance(job.exception, ApiError)
        self.assertStatus(job, JobStatus.ERROR)

    def test_status_flow_for_throwing_api(self):
        job = self.run_with_api(ThrowingAPI())
        self.assertStatus(job, JobStatus.INITIALIZING)

        self.wait_for_initialization(job)
        self.assertIsInstance(job.exception, ApiError)
        self.assertStatus(job, JobStatus.ERROR)

    def test_cancelled_result(self):
        job = self.run_with_api(CancellableAPI())

        self.wait_for_initialization(job)
        job.cancel()
        self._current_api.progress()
        self.assertEqual(job.result().get_status(), 'CANCELLED')
        self.assertStatus(job, JobStatus.CANCELLED)

    def test_errored_result(self):
        job = self.run_with_api(ThrowingInitializationAPI())

        # TODO: Seems inconsistent, should throw while initializating?
        self.wait_for_initialization(job)
        self.assertEqual(job.result().get_status(), 'ERROR')
        self.assertStatus(job, JobStatus.ERROR)

    def test_completed_result(self):
        job = self.run_with_api(NonQueuedAPI())

        self.wait_for_initialization(job)
        self._current_api.progress()
        self.assertEqual(job.result().get_status(), 'COMPLETED')
        self.assertStatus(job, JobStatus.DONE)

    def test_block_on_result_waiting_until_completed(self):
        from concurrent import futures

        job = self.run_with_api(NonQueuedAPI())
        with futures.ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.get_status(), 'COMPLETED')
        self.assertStatus(job, JobStatus.DONE)

    def test_block_on_result_waiting_until_cancelled(self):
        from concurrent.futures import ThreadPoolExecutor

        job = self.run_with_api(CancellableAPI())
        with ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.get_status(), 'CANCELLED')
        self.assertStatus(job, JobStatus.CANCELLED)

    def test_block_on_result_waiting_until_exception(self):
        from concurrent.futures import ThreadPoolExecutor
        job = self.run_with_api(ThrowingAPI())

        with ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.get_status(), 'ERROR')
        self.assertStatus(job, JobStatus.ERROR)

    def test_never_complete_result_with_timeout(self):
        job = self.run_with_api(NonQueuedAPI())

        self.wait_for_initialization(job)
        # We never make the API status to progress so it is stuck on RUNNING
        self.assertEqual(job.result(timeout=1).get_status(), 'ERROR')
        self.assertStatus(job, JobStatus.RUNNING)

    @unittest.expectedFailure
    def test_cancel_while_initializing_does_not_fail(self):
        job = self.run_with_api(CancellableAPI())
        job.cancel()
        self.assertStatus(job, JobStatus.CANCELLED)

    def test_only_final_states_cause_datailed_request(self):
        from unittest import mock

        # The state ERROR_CREATING_JOB is only handled when running the job,
        # and not while checking the status, so it is not tested.
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
                    _ = job.status
                    if status in API_FINAL_STATES:
                        self.assertTrue(self._current_api.get_job.called)
                    else:
                        self.assertFalse(self._current_api.get_job.called)

    def wait_for_initialization(self, job, timeout=1):
        """Waits until the job progress from `INITIALIZING` to a different
        status."""
        waited = 0
        wait = 0.1
        while job.status['status'] == JobStatus.INITIALIZING:
            time.sleep(wait)
            waited += wait
            if waited > timeout:
                self.fail(
                    msg="The JOB is still initializing after timeout ({}s)"
                    .format(timeout)
                )

    def assertStatus(self, job, status):
        """Assert the intenal job status is the expected one and also tests
        if the shorthand method for that status returns `True`."""
        self.assertEqual(job.status['status'], status)
        if status == JobStatus.CANCELLED:
            self.assertTrue(job.cancelled)
        elif status == JobStatus.DONE:
            self.assertTrue(job.done)
        elif status == JobStatus.VALIDATING:
            self.assertTrue(job.validating)
        elif status == JobStatus.RUNNING:
            self.assertTrue(job.running)
        elif status == JobStatus.QUEUED:
            self.assertTrue(job.queued)

    def run_with_api(self, api):
        """Creates a new `IBMQJob` instance running with the provided API
        object."""
        self._current_api = api
        self._current_qjob = IBMQJob(Qobj.from_dict(new_fake_qobj()), api,
                                     False)
        return self._current_qjob


def _auto_progress_api(api, interval=0.2):
    """Progress a `BaseFakeAPI` instacn every `interval` seconds until reaching
    the final state."""
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
        summary_fields = ['status', 'infoQueue']
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
    the job."""

    _job_status = [
        {'status': 'ERROR_CREATING_JOB'}
    ]


class ErrorWhileRunningAPI(BaseFakeAPI):
    """Class emulating an API processing a job that errors while running."""

    _job_status = [
        {'status': 'RUNNING'},
        {'status': 'ERROR_RUNNING_JOB'}
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


class ThrowingCancellationAPI(BaseFakeAPI):
    """Class for emulating an API with cancellation but throwing while
    trying."""

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
