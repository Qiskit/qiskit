# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

"""Test ExperimentData."""

import os
from unittest import mock
import copy
from random import randrange
import time
import threading
import json
import re

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeMelbourne
from qiskit.result import Result
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus
from qiskit.providers.experiment.experiment_data import ExperimentDataV1 as ExperimentData
from qiskit.providers.experiment.exceptions import ExperimentEntryExists
from qiskit.providers.experiment.experiment_service import ExperimentServiceV1
from qiskit.providers.experiment.exceptions import ExperimentError, ExperimentEntryNotFound


class TestExperimentData(QiskitTestCase):
    """Test the ExperimentData class."""

    def setUp(self):
        super().setUp()
        self.backend = FakeMelbourne()

    def test_experiment_data_attributes(self):
        """Test experiment data attributes."""
        attrs = {"job_ids": ['job1'],
                 "share_level": "global",
                 "figure_names": ['figure1'],
                 "notes": "some notes"}
        exp_data = ExperimentData(
            backend=self.backend,
            experiment_type='my_type',
            experiment_id='1234',
            tags=['tag1', 'tag2'],
            metadata={'foo': 'bar'},
            **attrs
        )
        self.assertEqual(exp_data.backend.name(), self.backend.name())
        self.assertEqual(exp_data.experiment_type, 'my_type')
        self.assertEqual(exp_data.experiment_id, '1234')
        self.assertEqual(exp_data.tags(), ['tag1', 'tag2'])
        self.assertEqual(exp_data.metadata(), {'foo': 'bar'})
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
        exp_data.block_for_jobs()
        self.assertEqual(expected, [sdata['counts'] for sdata in exp_data.data()])

    def test_add_data_job_callback(self):
        """Test add job data with callback."""
        def _callback(_exp_data):
            self.assertIsInstance(_exp_data, ExperimentData)
            self.assertEqual([dat['counts'] for dat in _exp_data.data()],
                             a_job.result().get_counts())
            nonlocal called_back
            called_back = True

        a_job = mock.create_autospec(Job, instance=True)
        a_job.result.return_value = self._get_job_result(2)

        called_back = False
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.add_data(a_job, post_processing_callback=_callback)
        exp_data.block_for_jobs()
        self.assertTrue(called_back)

    def test_get_data(self):
        """Test getting data."""
        data1 = []
        for _ in range(5):
            data1.append({'counts': {'00': randrange(1024)}})
        results = self._get_job_result(3)

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(data1)
        exp_data.add_data(results)
        self.assertEqual(data1[1], exp_data.data(1))
        self.assertEqual(data1[2:4], exp_data.data(slice(2, 4)))
        self.assertEqual(results.get_counts(),
                         [sdata['counts'] for sdata in exp_data.data(results.job_id)])

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
                fn = exp_data.add_figure(figure, figure_name)
                self.assertEqual(hello_bytes, exp_data.figure(fn))

    def test_add_figure_overwrite(self):
        """Test updating an existing figure."""
        hello_bytes = str.encode("hello world")
        friend_bytes = str.encode("hello friend!")

        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        fn = exp_data.add_figure(hello_bytes)
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

    def test_delete_figure(self):
        """Test deleting a figure."""
        exp_data = ExperimentData(experiment_type='my_type')
        id_template = "figure_{}"
        for idx in range(3):
            exp_data.add_figure(str.encode("hello world"), id_template.format(idx))

        sub_tests = [(1, id_template.format(1)),
                     (id_template.format(2), id_template.format(2))]

        for del_key, figure_name in sub_tests:
            with self.subTest(del_key=del_key):
                exp_data.delete_figure(del_key)
                self.assertRaises(ExperimentEntryNotFound, exp_data.figure, figure_name)

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

    def test_delete_analysis_result(self):
        """Test deleting analysis result."""
        exp_data = ExperimentData(experiment_type='my_type')
        id_template = "result_{}"
        for idx in range(3):
            res = mock.MagicMock()
            res.result_id = id_template.format(idx)
            exp_data.add_analysis_result(res)

        subtests = [(0, id_template.format(0)),
                    (id_template.format(2), id_template.format(2))]
        for del_key, res_id in subtests:
            with self.subTest(del_key=del_key):
                exp_data.delete_analysis_result(del_key)
                self.assertRaises(ExperimentEntryNotFound, exp_data.analysis_result, res_id)

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

    def test_save_all(self):
        """Test saving all experiment related data."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        service = mock.create_autospec(ExperimentServiceV1, instance=True)
        exp_data.add_figure(str.encode("hello world"))
        analysis_result = mock.MagicMock()
        exp_data.add_analysis_result(analysis_result)
        exp_data.save_all(service=service)
        service.create_experiment.assert_called_once()
        service.create_figure.assert_called_once()
        analysis_result.save.assert_called_once()

    def test_save_all_delete(self):
        """Test saving all deletion."""
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        service = mock.create_autospec(ExperimentServiceV1, instance=True)
        exp_data.add_figure(str.encode("hello world"))
        exp_data.add_analysis_result(mock.MagicMock())
        exp_data.delete_analysis_result(0)
        exp_data.delete_figure(0)

        exp_data.save_all(service=service)
        service.create_experiment.assert_called_once()
        service.delete_figure.assert_called_once()
        service.delete_analysis_result.assert_called_once()

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

    def test_auto_save(self):
        """Test auto save."""
        service = self._set_mock_service()
        exp_data = ExperimentData(backend=self.backend, experiment_type='my_type')
        exp_data.auto_save = True

        mock_result = mock.MagicMock()
        exp_data.save()

        subtests = [
            # update function, update parameters, service called
            (exp_data.add_analysis_result, (mock_result,), mock_result.save),
            (exp_data.add_figure, (str.encode("hello world"),), service.create_figure),
            (exp_data.delete_figure, (0,), service.delete_figure),
            (exp_data.delete_analysis_result, (0,), service.delete_analysis_result),
            (exp_data.update_tags, (['foo'],), None),
            (exp_data.update_metadata, ({'foo': 'bar'},), None),
            (setattr, (exp_data, 'notes', 'foo'), None)
        ]

        for func, params, called in subtests:
            with self.subTest(func=func):
                func(*params)
                service.update_experiment.assert_called_once()
                if called:
                    called.assert_called_once()
                service.reset_mock()

    def test_status_job_pending(self):
        """Test experiment status when job is pending."""
        job1 = mock.create_autospec(Job, instance=True)
        job1.result.return_value = self._get_job_result(3)
        job1.status.return_value = JobStatus.DONE

        event = threading.Event()
        job2 = mock.create_autospec(Job, instance=True)
        job2.result = lambda *args, **kwargs: event.wait()
        job2.status.return_value = JobStatus.RUNNING
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job1)
        exp_data.add_data(job2, lambda *args, **kwargs: event.wait())
        self.assertEqual('RUNNING', exp_data.status())

    def test_status_job_error(self):
        """Test experiment status when job failed."""
        job1 = mock.create_autospec(Job, instance=True)
        job1.result.return_value = self._get_job_result(3)
        job1.status.return_value = JobStatus.DONE

        job2 = mock.create_autospec(Job, instance=True)
        job2.status.return_value = JobStatus.ERROR

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data([job1, job2])
        self.assertEqual('ERROR', exp_data.status())

    def test_status_post_processing(self):
        """Test experiment status during post processing."""
        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)

        event = threading.Event()
        self.addCleanup(event.set)

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job)
        exp_data.add_data(job, lambda *args, **kwargs: event.wait())
        self.assertEqual('POST_PROCESSING', exp_data.status())

    def test_status_post_processing_error(self):
        """Test experiment status when post processing failed."""
        def _post_processing(*args, **kwargs):
            raise ValueError("Kaboom!")

        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job)
        exp_data.add_data(job, _post_processing)
        exp_data.block_for_jobs()
        self.assertEqual('ERROR', exp_data.status())

    def test_status_done(self):
        """Test experiment status when all jobs are done."""
        job = mock.create_autospec(Job, instance=True)
        job.result.return_value = self._get_job_result(3)
        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job)
        exp_data.add_data(job, lambda *args, **kwargs: time.sleep(1))
        exp_data.block_for_jobs()
        self.assertEqual('DONE', exp_data.status())

    def test_update_tags(self):
        """Test updating experiment tags."""
        exp_data = ExperimentData(experiment_type='my_type', tags=['foo'])
        self.assertEqual(['foo'], exp_data.tags())
        exp_data.update_tags(['bar'])
        self.assertEqual(['bar'], exp_data.tags())

    def test_update_metadata(self):
        """Test updating experiment metadata."""
        exp_data = ExperimentData(experiment_type='my_type', metadata={'foo': 'bar'})
        self.assertEqual({'foo': 'bar'}, exp_data.metadata())
        exp_data.update_metadata({'bar': 'foo'})
        self.assertEqual({'bar': 'foo'}, exp_data.metadata())

    def test_cancel_jobs(self):
        """Test canceling experiment jobs."""
        exp_data = ExperimentData(experiment_type='my_type')
        event = threading.Event()
        self.addCleanup(event.set)
        job = mock.create_autospec(Job, instance=True)
        job.result = lambda *args, **kwargs: event.wait()
        exp_data.add_data(job)
        exp_data.cancel_jobs()
        job.cancel.assert_called_once()

    def test_metadata_serialization(self):
        """Test experiment metadata serialization."""
        metadata = {'complex': 2+3j, 'numpy': np.zeros(2)}
        exp_data = ExperimentData(experiment_type='my_type',
                                  metadata=metadata)
        serialized = exp_data._serialize_metadata()
        self.assertIsInstance(serialized, str)
        self.assertTrue(json.loads(serialized))

        deserialized = ExperimentData.deserialize_metadata(serialized)
        self.assertEqual(metadata['complex'], deserialized['complex'])
        self.assertEqual(metadata['numpy'].all(), deserialized['numpy'].all())

    def test_errors(self):
        """Test getting experiment error message."""
        def _post_processing(*args, **kwargs):  # pylint: disable=unused-argument
            raise ValueError("Kaboom!")

        job1 = mock.create_autospec(Job, instance=True)
        job1.job_id.return_value = '1234'

        job2 = mock.create_autospec(Job, instance=True)
        job2.status.return_value = JobStatus.ERROR
        job2.job_id.return_value = '5678'

        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job1, _post_processing)
        exp_data.add_data(job2)
        exp_data.block_for_jobs()
        self.assertEqual('ERROR', exp_data.status())
        self.assertTrue(re.match(r".*1234.*Kaboom!.*5678", exp_data.errors(), re.DOTALL))

    def test_source(self):
        """Test getting experiment source."""
        exp_data = ExperimentData(experiment_type='my_type')
        source_vals = '\n'.join([str(val) for val in exp_data.source.values()])
        self.assertIn('ExperimentDataV1', source_vals)
        self.assertIn('qiskit-terra', source_vals)

    def test_block_for_jobs(self):
        """Test blocking for jobs."""
        def _sleeper(*args, **kwargs):  # pylint: disable=unused-argument
            time.sleep(2)
            nonlocal sleep_count
            sleep_count += 1
            return self._get_job_result(1)

        sleep_count = 0
        job = mock.create_autospec(Job, instance=True)
        job.result = _sleeper
        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(job, _sleeper)
        exp_data.block_for_jobs()
        self.assertEqual(2, sleep_count)

    def test_additional_attr(self):
        """Test additional experiment attributes."""
        exp_data = ExperimentData(experiment_type='my_type', foo='foo')
        self.assertEqual('foo', exp_data.foo)

    def test_str(self):
        """Test the string representation."""
        exp_data = ExperimentData(experiment_type='my_type')
        exp_data.add_data(self._get_job_result(1))
        result = mock.MagicMock()
        exp_data.add_analysis_result(result)
        exp_data_str = str(exp_data)
        self.assertIn(exp_data.experiment_type, exp_data_str)
        self.assertIn(exp_data.experiment_id, exp_data_str)
        self.assertIn(str(result), exp_data_str)

    def _get_job_result(self, circ_count, has_metadata=False):
        """Return a job result with random counts."""
        job_result = {
            'backend_name': self.backend.name(),
            'backend_version': '1.1.1',
            'qobj_id': '1234',
            'job_id': 'some_job_id',
            'success': True,
            'results': []}
        circ_result_template = {
            'shots': 1024,
            'success': True,
            'data': {}
        }

        for _ in range(circ_count):
            counts = randrange(1024)
            circ_result = copy.copy(circ_result_template)
            circ_result['data'] = {'counts': {'0x0': counts, '0x3': 1024-counts}}
            if has_metadata:
                circ_result['header'] = {"metadata": {"meas_basis": "pauli"}}
            job_result['results'].append(circ_result)

        return Result.from_dict(job_result)

    def _set_mock_service(self):
        """Add a mock service to the backend."""
        mock_provider = mock.MagicMock()
        self.backend._provider = mock_provider
        mock_service = mock.create_autospec(ExperimentServiceV1, instance=True)
        mock_provider.service.return_value = mock_service
        return mock_service
