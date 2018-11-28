# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except


"""IBMQJob states test-suite."""

import unittest
import time
from contextlib import suppress
from qiskit.backends.jobstatus import JobStatus
from qiskit.backends.ibmq.ibmqjob import IBMQJobPreQobj, IBMQJob, API_FINAL_STATES
from qiskit.backends.ibmq.api import ApiError
from qiskit.backends import JobError, JobTimeoutError
from ..common import JobTestCase
from .._mockutils import new_fake_qobj, FakeBackend


class TestIBMQJobStates(JobTestCase):
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

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.VALIDATING)

    def test_error_while_creating_job(self):
        job = self.run_with_api(ErrorWhileCreatingAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_error_while_validating_job(self):
        job = self.run_with_api(ErrorWhileValidatingAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.VALIDATING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_status_flow_for_non_queued_job(self):
        job = self.run_with_api(NonQueuedAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_status_flow_for_queued_job(self):
        job = self.run_with_api(QueuedAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.QUEUED)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_status_flow_for_cancellable_job(self):
        job = self.run_with_api(CancellableAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertTrue(can_cancel)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.CANCELLED)

    def test_status_flow_for_non_cancellable_job(self):
        job = self.run_with_api(NonCancellableAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        can_cancel = job.cancel()
        self.assertFalse(can_cancel)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.RUNNING)

    def test_status_flow_for_errored_cancellation(self):
        job = self.run_with_api(ErroredCancellationAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)
        can_cancel = job.cancel()
        self.assertFalse(can_cancel)
        self.assertEqual(job.status(), JobStatus.RUNNING)

    def test_status_flow_for_unable_to_run_valid_qobj(self):
        """Contrary to other tests, this one is expected to fail even for a
        non-job-related issue. If the API fails while sending a job, we don't
        get an id so we can not query for the job status."""
        job = self.run_with_api(UnavailableRunAPI())

        with self.assertRaises(JobError):
            self.wait_for_initialization(job)

        with self.assertRaises(JobError):
            job.status()

    def test_api_throws_temporarily_but_job_is_finished(self):
        job = self.run_with_api(ThrowingNonJobRelatedErrorAPI(errors_before_success=2))

        # First time we query the server...
        with self.assertRaises(JobError):
            # The error happens inside wait_for_initialization, the first time
            # it calls to status() after INITIALIZING.
            self.wait_for_initialization(job)

        # Also an explicit second time...
        with self.assertRaises(JobError):
            job.status()

        # Now the API gets fixed and doesn't throw anymore.
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_status_flow_for_unable_to_run_invalid_qobj(self):
        job = self.run_with_api(RejectingJobAPI())
        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.ERROR)

    def test_error_while_running_job(self):
        job = self.run_with_api(ErrorWhileRunningAPI())

        self.wait_for_initialization(job)
        self.assertEqual(job.status(), JobStatus.RUNNING)

        self._current_api.progress()
        self.assertEqual(job.status(), JobStatus.ERROR)
        self.assertEqual(job.error_message(), 'Error running job')

    def test_cancelled_result(self):
        job = self.run_with_api(CancellableAPI())

        self.wait_for_initialization(job)
        job.cancel()
        self._current_api.progress()
        with self.assertRaises(JobError):
            _ = job.result()
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
        self.assertEqual(job.result().success, True)
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_block_on_result_waiting_until_completed(self):
        from concurrent import futures

        job = self.run_with_api(NonQueuedAPI())
        with futures.ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        result = job.result()
        self.assertEqual(result.success, True)
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_block_on_result_waiting_until_cancelled(self):
        from concurrent.futures import ThreadPoolExecutor

        job = self.run_with_api(CancellableAPI())
        with ThreadPoolExecutor() as executor:
            executor.submit(_auto_progress_api, self._current_api)

        with self.assertRaises(JobError):
            job.result()

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
        with self.assertRaises(JobTimeoutError):
            job.result(timeout=0.2)

    def test_cancel_while_initializing_fails(self):
        job = self.run_with_api(CancellableAPI())
        can_cancel = job.cancel()
        self.assertFalse(can_cancel)
        self.assertEqual(job.status(), JobStatus.INITIALIZING)

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

                with suppress(BaseFakeAPI.NoMoreStatesError):
                    self._current_api.progress()

                with mock.patch.object(self._current_api, 'get_job',
                                       wraps=self._current_api.get_job):
                    job.status()
                    if status in API_FINAL_STATES:
                        self.assertTrue(self._current_api.get_job.called)
                    else:
                        self.assertFalse(self._current_api.get_job.called)

    # TODO: Once qobj results come by default from all the simulator
    # backends, move to integration tests in test_result.py
    def test_qobj_result(self):
        job = self.run_with_api(QObjResultAPI(), job_class=IBMQJob)

        self.wait_for_initialization(job)
        self._current_api.progress()
        result = job.result()
        self.assertEqual(result.success, True)
        self.assertEqual(result.get_counts('Bell state'),
                         {'0x0': 480, '0x3': 490, '0x1': 20, '0x2': 34})
        self.assertEqual(result.get_counts('Bell state XY'),
                         {'0x0': 29, '0x3': 15, '0x1': 510, '0x2': 480})
        self.assertEqual(len(result.results), 2)

    def run_with_api(self, api, job_class=IBMQJobPreQobj):
        """Creates a new ``IBMQJobPreQobj`` instance running with the provided API
        object.
        """
        backend = FakeBackend()
        self._current_api = api
        self._current_qjob = job_class(backend, None, api, False, qobj=new_fake_qobj())
        self._current_qjob.submit()
        return self._current_qjob


def _auto_progress_api(api, interval=0.2):
    """Progress a `BaseFakeAPI` instacn every `interval` seconds until reaching
    the final state.
    """
    with suppress(BaseFakeAPI.NoMoreStatesError):
        while True:
            time.sleep(interval)
            api.progress()


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


class RejectingJobAPI(BaseFakeAPI):
    """Class for emulating an API unable of initializing."""

    def run_job(self, *_args, **_kwargs):
        return {'error': 'invalid qobj'}


class UnavailableRunAPI(BaseFakeAPI):
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


class ThrowingNonJobRelatedErrorAPI(BaseFakeAPI):
    """Class for emulating an scenario where the job is done but the API
    fails some times for non job-related errors.
    """

    _job_status = [
        {'status': 'COMPLETED'}
    ]

    def __init__(self, errors_before_success=2):
        super().__init__()
        self._number_of_exceptions_to_throw = errors_before_success

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
        raise ApiError('Unexpected error')


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


# TODO: Remove once qobj results come by default from all the simulator
# backends.
class QObjResultAPI(BaseFakeAPI):
    """Class for emulating a successfully-completed non-queued API."""

    _job_status = [
        {'status': 'RUNNING'},
        {
            'status': 'COMPLETED',
            'qObjectResult': {
                'backend_name': 'ibmqx2',
                'backend_version': '1.1.1',
                'job_id': 'XC1323XG2',
                'qobj_id': 'Experiment1',
                'success': True,
                'status': 'COMPLETED',
                'results': [
                    {
                        'header': {'name': 'Bell state'},
                        'shots': 1024,
                        'status': 'DONE',
                        'success': True,
                        'data': {
                            'counts': {
                                '0x0': 480, '0x3': 490, '0x1': 20, '0x2': 34
                            }
                        }
                    },
                    {
                        'header': {'name': 'Bell state XY'},
                        'shots': 1024,
                        'status': 'DONE',
                        'success': True,
                        'data': {
                            'counts': {
                                '0x0': 29, '0x3': 15, '0x1': 510, '0x2': 480
                            }
                        }
                    }
                ]
            }
        }
    ]


if __name__ == '__main__':
    unittest.main(verbosity=2)
