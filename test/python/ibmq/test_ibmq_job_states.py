# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring


"""IBMQJob states test-suite."""

import unittest
from contextlib import suppress
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.ibmq.ibmqjob import IBMQJobPreQobj, API_FINAL_STATES
from qiskit.providers import JobError, JobTimeoutError
from qiskit.test.mock import (FakeRueschlikon, BaseFakeAPI, UnknownStatusAPI, ValidatingAPI,
                              ErrorWhileValidatingAPI, NonQueuedAPI, ErrorWhileCreatingAPI,
                              ErrorWhileRunningAPI, QueuedAPI, RejectingJobAPI, UnavailableRunAPI,
                              ThrowingAPI, ThrowingNonJobRelatedErrorAPI, ThrowingGetJobAPI,
                              CancellableAPI, NonCancellableAPI, ErroredCancellationAPI,
                              new_fake_qobj, _auto_progress_api)
from .jobtestcase import JobTestCase


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

    def run_with_api(self, api, job_class=IBMQJobPreQobj):
        """Creates a new ``IBMQJobPreQobj`` instance running with the provided API
        object.
        """
        backend = FakeRueschlikon()
        self._current_api = api
        self._current_qjob = job_class(backend, None, api, False, qobj=new_fake_qobj())
        self._current_qjob.submit()
        return self._current_qjob


if __name__ == '__main__':
    unittest.main(verbosity=2)
