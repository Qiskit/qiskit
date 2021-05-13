# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

"""Test experiment methods."""

import os
import unittest
from unittest import mock
import copy
from random import randrange
import time

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.result import Result
from qiskit.providers import JobV1 as Job
from qiskit.providers.experiment.experiment_data import ExperimentDataV1 as ExperimentData
from qiskit.providers.experiment.exceptions import ExperimentEntryExists
from qiskit.providers.experiment.experiment_service import ExperimentServiceV1
from qiskit.providers.experiment.exceptions import ExperimentError


class TestExperimentData(unittest.TestCase):
    """Test the ExperimentData class."""

    def setUp(self):
        super().setUp()
        self.backend = FakeMelbourne()

    def test_experiment_data_attributes(self):
        """Test experiment data attributes."""
        attrs = {"job_ids": ['job1'],
                 "share_level": "global",
                 "metadata": {'foo': 'bar'},
                 "figure_names": ['figure1'],
                 "notes": "some notes"}
        exp_data = ExperimentData(
            backend=self.backend,
            experiment_type='my_type',
            experiment_id='1234',
            tags=['tag1', 'tag2'],
            **attrs
        )
        self.assertEqual(exp_data.backend.name(), self.backend.name())
        self.assertEqual(exp_data.experiment_type, 'my_type')
        self.assertEqual(exp_data.experiment_id, '1234')
        self.assertEqual(exp_data.tags(), ['tag1', 'tag2'])
        for key, val in attrs.items():
            self.assertEqual(getattr(exp_data, key), val)

    def test_add_data_dict(self):
        """Test add data in dictionary."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        a_dict = {'counts': {'01': 518}}
        dicts = [{'counts': {'00': 284}}, {'counts': {'00': 14}}]

        exp_data.add_data(a_dict)
        exp_data.add_data(dicts)
        self.assertEqual([a_dict] + dicts, exp_data.data())

    def test_add_data_result(self):
        """Test add result data."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        a_result = self._get_job_result(1)
        results = [self._get_job_result(2), self._get_job_result(3)]

        expected = [a_result.get_counts()]
        for res in results:
            expected.extend(res.get_counts())

        exp_data.add_data(a_result)
        exp_data.add_data(results)
        self.assertEqual(expected, [sdata['counts'] for sdata in exp_data.data()])

    def test_add_data_result_metadata(self):
        """Test add result metadata."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        result1 = self._get_job_result(1, has_metadata=False)
        result2 = self._get_job_result(1, has_metadata=True)

        exp_data.add_data(result1)
        exp_data.add_data(result2)
        self.assertNotIn('metadata', exp_data.data(0))
        self.assertIn('metadata', exp_data.data(1))

    def _get_job_result(self, circ_count, has_metadata=False):
        """Return a job result with random counts."""
        job_result = {
            'backend_name': self.backend.name(),
            'backend_version': '1.1.1',
            'qobj_id': '1234',
            'job_id': '5678',
            'success': True,
            'results': []}
        circ_result_template = {
            'shots': 1024,
            'success': True,
            'data': {}
        }

        for i in range(circ_count):
            counts = randrange(1024)
            circ_result = copy.copy(circ_result_template)
            circ_result['data'] = {'counts': {'0x0': counts, '0x3': 1024-counts}}
            if has_metadata:
                circ_result['header'] = {"metadata": {"meas_basis": "pauli"}}
            job_result['results'].append(circ_result)

        return Result.from_dict(job_result)

    def test_add_data_job(self):
        """Test add job data."""
        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(3)
        jobs = []
        for _ in range(2):
            job = mock.create_autospec(Job, instance=True)
            job.result.return_value = self._get_job_result(2)
            jobs.append(job)

        expected = a_job.result().get_counts()
        for job in jobs:
            expected.extend(job.result().get_counts())

        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_data(a_job)
        exp_data.add_data(jobs)
        self.assertEqual(expected, [sdata['counts'] for sdata in exp_data.data()])

    def test_add_data_job_callback(self):
        """Test add job data with callback."""
        def _callback(_exp_data):
            try:
                self.assertIsInstance(_exp_data, ExperimentData)
                self.assertEqual([dat['counts'] for dat in _exp_data.data()],
                                 a_job.result().get_counts())
            except AssertionError as err:
                # self.log.error(f"{type(err)}: {err}")
                raise
            nonlocal called_back
            called_back = True

        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(2)

        called_back = False
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_data(a_job, post_processing_callback=_callback)
        for _ in range(3):
            if called_back:
                break
            time.sleep(1)
        self.assertTrue(called_back)

    def test_add_data_jobs_delay(self):
        """Test add job data with delays getting results."""
        def _result(_delay, _result):
            def _wrapped(*args, **kwargs):
                time.sleep(_delay)
                return _result
            return _wrapped

        jobs = []
        expected = []
        max_delay = 3
        for idx in range(max_delay):
            job = mock.create_autospec(Job, instance=True)
            result = self._get_job_result(2)
            expected.extend(result.get_counts())
            job.result = _result(idx, result)
            jobs.append(job)

        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_data(jobs)
        time.sleep(max_delay+1)  # Wait for all jobs to finish
        self.assertEqual(expected, [sdata['counts'] for sdata in exp_data.data()])

    def test_get_data(self):
        """Test getting data."""
        data1 = []
        for _ in range(5):
            data1.append({'counts': {'00': randrange(1024)}})

        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)

        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_data(data1)
        exp_data.add_data(job)
        self.assertEqual(data1[1], exp_data.data(1))
        self.assertEqual(data1[2:4], exp_data.data(slice(2, 4)))
        self.assertEqual(job.result().get_counts(),
                         [sdata['counts'] for sdata in exp_data.data(job.result().job_id)])

    def test_add_figure(self):
        """Test adding a new figure."""
        hello_bytes = str.encode("hello world")
        file_name = "hello_world.svg"
        self.addCleanup(os.remove, file_name)
        with open(file_name, 'wb') as file:
            file.write(hello_bytes)

        sub_tests = [('file name', file_name, None),
                     ('file bytes', hello_bytes, None),
                     ('new name', hello_bytes, 'hello_again.svg')]

        for name, figure, figure_name in sub_tests:
            with self.subTest(name=name):
                exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
                fn, size = exp_data.add_figure(figure, figure_name)
                self.assertEqual(hello_bytes, exp_data.figure(fn))

    def test_add_figure_overwrite(self):
        """Test updating an existing figure."""
        hello_bytes = str.encode("hello world")
        friend_bytes = str.encode("hello friend!")

        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        fn, size = exp_data.add_figure(hello_bytes)
        with self.assertRaises(ExperimentEntryExists):
            exp_data.add_figure(friend_bytes, fn)

        exp_data.add_figure(friend_bytes, fn, overwrite=True)
        self.assertEqual(friend_bytes, exp_data.figure(fn))

    def test_get_figure(self):
        """Test getting figure."""
        exp_data = ExperimentData(experiment_type='my_type')
        figure_template = "hello world {}"
        name_template = "figure_{}"
        for idx in range(3):
            exp_data.add_figure(str.encode(figure_template.format(idx)),
                                figure_name=name_template.format(idx))
        idx = randrange(3)
        expected_figure = str.encode(figure_template.format(idx))
        self.assertEqual(expected_figure, exp_data.figure(name_template.format(idx)))
        self.assertEqual(expected_figure, exp_data.figure(idx))

        file_name = "hello_world.svg"
        self.addCleanup(os.remove, file_name)
        exp_data.figure(idx, file_name)
        with open(file_name, 'rb') as file:
            self.assertEqual(expected_figure, file.read())

    def test_save_figure(self):
        """Test saving figure when adding it."""
        mock_service = self._set_mock_service()
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_figure(str.encode("hello world"), figure_name="hello", save=True)
        mock_service.create_figure.assert_called_once()
        new_service = mock.MagicMock()
        exp_data.add_figure(str.encode("hello world"), figure_name="hello", overwrite=True,
                            save=True, service=new_service)
        new_service.update_figure.assert_called_once()

    def test_delayed_backend(self):
        """Test initializing experiment data without a backend."""
        exp_data = ExperimentData(experiment_type='my_type')
        self.assertIsNone(exp_data.backend)
        self.assertIsNone(exp_data.service)
        exp_data.save()
        a_job = mock.create_autospec(Job, instance=True)
        exp_data.add_data(a_job)
        self.assertIsNotNone(exp_data.backend)
        self.assertIsNotNone(exp_data.service)

    def test_different_backend(self):
        """Test setting a different backend."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        a_job = mock.create_autospec(Job, instance=True)
        self.assertNotEqual(exp_data.backend, a_job.backend())
        with self.assertLogs('qiskit.providers.experiment', 'WARNING'):
            exp_data.add_data(a_job)

    def test_add_get_analysis_result(self):
        """Test adding and getting analysis results."""
        exp_data = ExperimentData(experiment_type='my_type')
        results = []
        for idx in range(5):
            res = mock.MagicMock()
            res.result_id = idx
            results.append(res)
            exp_data.add_analysis_result(res)

        self.assertEqual(results, exp_data.analysis_result())
        self.assertEqual(results[1], exp_data.analysis_result(1))
        self.assertEqual(results[2:4], exp_data.analysis_result(slice(2, 4)))
        self.assertEqual(results[4], exp_data.analysis_result(results[4].result_id))

    def test_save(self):
        """Test saving experiment data."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        service = mock.create_autospec(ExperimentServiceV1, instance=True)
        exp_data.save(service=service)
        service.create_experiment.assert_called_once()
        self.assertEqual(exp_data.experiment_id,
                         service.create_experiment.call_args.kwargs['experiment_id'])
        exp_data.save(service=service)
        service.update_experiment.assert_called_once()
        self.assertEqual(exp_data.experiment_id,
                         service.update_experiment.call_args.kwargs['experiment_id'])

    def test_set_service_backend(self):
        """Test setting service via backend."""
        mock_service = self._set_mock_service()
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        self.assertEqual(mock_service, exp_data.service)

    def test_set_service_job(self):
        """Test setting service via adding a job."""
        mock_service = self._set_mock_service()
        job = mock.create_autospec(Job, instance=True)
        job.backend.return_value = self.backend
        exp_data = ExperimentData(experiment_type='my_type')
        self.assertIsNone(exp_data.service)
        exp_data.add_data(job)
        self.assertEqual(mock_service, exp_data.service)

    def test_set_service_direct(self):
        """Test setting service directly."""
        exp_data = ExperimentData(experiment_type='my_type')
        self.assertIsNone(exp_data.service)
        mock_service = mock.MagicMock()
        exp_data.service = mock_service
        self.assertEqual(mock_service, exp_data.service)

        with self.assertRaises(ExperimentError):
            exp_data.service = mock_service

    def test_set_service_save(self):
        """Test setting service when saving."""
        orig_service = self._set_mock_service()
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        new_service = mock.create_autospec(ExperimentServiceV1, instance=True)
        exp_data.save(service=new_service)
        new_service.create_experiment.assert_called()
        orig_service.create_experiment.assert_not_called()

    def test_new_backend_has_service(self):
        """Test changing backend doesn't change existing service."""
        orig_service = self._set_mock_service()
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        self.assertEqual(orig_service, exp_data.service)

        job = mock.create_autospec(Job, instance=True)
        new_service = self._set_mock_service()
        self.assertNotEqual(orig_service, new_service)
        job.backend.return_value = self.backend
        exp_data.add_data(job)
        self.assertEqual(orig_service, exp_data.service)

    def _set_mock_service(self):
        mock_provider = mock.MagicMock()
        self.backend._provider = mock_provider
        mock_service = mock.create_autospec(ExperimentServiceV1, instance=True)
        mock_provider.service.return_value = mock_service
        return mock_service

    def test_auto_save(self):
        # 1. add_data
        # 2. add_analysis_result
        # 3. update_tags
        pass

    def test_status(self):
        pass

    def test_tags(self):
        pass

    def test_retrieved_experiment(self):
        pass

    def test_cancel_jobs(self):
        pass

    def test_metadata_serialization(self):
        pass

    def test_errors(self):
        pass

    def test_source(self):
        pass


class TestAnalysisResult(QiskitTestCase):

    def setUp(self):
        super().setUp()
        self.backend = FakeMelbourne()

    def test_analysis_result_attributes(self):
        pass

    def test_save(self):
        pass

    def test_auto_save(self):
        pass

    def test_update_data(self):
        pass
