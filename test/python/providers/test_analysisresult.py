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

"""Test AnalysisResult."""

from datetime import datetime
from unittest import mock
import json

import numpy as np
from dateutil import tz

from qiskit.test import QiskitTestCase
from qiskit.providers.experiment.analysis_result import AnalysisResultV1 as AnalysisResult
from qiskit.providers.experiment.device_component import Qubit, Resonator, to_component
from qiskit.providers.experiment.constants import ResultQuality
from qiskit.providers.experiment.experiment_service import ExperimentServiceV1
from qiskit.providers.experiment.exceptions import ExperimentError


class TestAnalysisResult(QiskitTestCase):
    """Test the AnalysisResult class."""

    def test_analysis_result_attributes(self):
        """Test analysis result attributes."""
        attrs = {"result_type": 'my_type',
                 "device_components": [Qubit(1), Qubit(2)],
                 "experiment_id": "1234",
                 "result_id": "5678",
                 "quality": ResultQuality.GOOD,
                 "verified": False}
        result = AnalysisResult(
            result_data={'foo': 'bar'},
            tags=['tag1', 'tag2'],
            **attrs
        )
        self.assertEqual({'foo': 'bar'}, result.data())
        self.assertEqual(['tag1', 'tag2'], result.tags())
        for key, val in attrs.items():
            self.assertEqual(val, getattr(result, key))

    def test_save(self):
        """Test saving analysis result."""
        mock_service = mock.create_autospec(ExperimentServiceV1)
        result = self._new_analysis_result()
        result.save(service=mock_service)
        mock_service.create_analysis_result.assert_called_once()

    def test_auto_save(self):
        """Test auto saving."""
        mock_service = mock.create_autospec(ExperimentServiceV1)
        result = self._new_analysis_result(service=mock_service)
        result.auto_save = True
        result.save()

        subtests = [
            # update function, update parameters, service called
            (result.update_tags, (['foo'],)),
            (result.update_data, ({'foo': 'bar'},)),
            (setattr, (result, 'quality', 'GOOD')),
            (setattr, (result, 'verified', True))
        ]

        for func, params in subtests:
            with self.subTest(func=func):
                func(*params)
                mock_service.update_analysis_result.assert_called_once()
                mock_service.reset_mock()

    def test_set_service_init(self):
        """Test setting service in init."""
        mock_service = mock.create_autospec(ExperimentServiceV1)
        result = self._new_analysis_result(service=mock_service)
        self.assertEqual(mock_service, result.service)

    def test_set_service_direct(self):
        """Test setting service directly."""
        mock_service = mock.create_autospec(ExperimentServiceV1)
        result = self._new_analysis_result()
        result.service = mock_service
        self.assertEqual(mock_service, result.service)

        with self.assertRaises(ExperimentError):
            result.service = mock_service

    def test_set_service_save(self):
        """Test setting service when saving."""
        orig_service = mock.create_autospec(ExperimentServiceV1)
        result = self._new_analysis_result(service=orig_service)
        new_service = mock.create_autospec(ExperimentServiceV1)
        result.save(service=new_service)
        new_service.create_analysis_result.assert_called()
        orig_service.create_analysis_result.assert_not_called()

    def test_update_data(self):
        """Test updating data."""
        result = self._new_analysis_result()
        result.update_data({'foo': 'new data'})
        self.assertEqual({'foo': 'new data'}, result.data())

    def test_update_tags(self):
        """Test updating tags."""
        result = self._new_analysis_result()
        result.update_tags(['new_tag'])
        self.assertEqual(['new_tag'], result.tags())

    def test_update_quality(self):
        """Test updating quality."""
        result = self._new_analysis_result(quality='BAD')
        result.quality = 'GOOD'
        self.assertEqual(ResultQuality.GOOD, result.quality)

    def test_update_verified(self):
        """Test updating verified."""
        result = self._new_analysis_result(verified=False)
        result.verified = True
        self.assertTrue(result.verified)

    def test_additional_attr(self):
        """Test additional attributes."""
        result = self._new_analysis_result(foo='bar')
        self.assertEqual('bar', result.foo)

    def test_data_serialization(self):
        """Test result data serialization."""
        result = self._new_analysis_result(result_data={'complex': 2+3j,
                                                        'numpy': np.zeros(2)})
        serialized = result._serialize_data()
        self.assertIsInstance(serialized, str)
        self.assertTrue(json.loads(serialized))

    def test_creation_date(self):
        """Test creation date."""
        local_dt = datetime.now(tz=tz.tzlocal())
        result = self._new_analysis_result(creation_date=local_dt.astimezone(tz.UTC).isoformat())
        self.assertEqual(local_dt, result.creation_date)

    def test_source(self):
        """Test getting analysis result source."""
        result = self._new_analysis_result()
        source_vals = '\n'.join([str(val) for val in result.source.values()])
        self.assertIn('AnalysisResultV1', source_vals)
        self.assertIn('qiskit-terra', source_vals)

    def _new_analysis_result(self, **kwargs):
        """Return a new analysis result."""
        values = {'result_data': {'foo': 'bar'},
                  'result_type': 'some_type',
                  'device_components': ['Q1', 'Q1'],
                  'experiment_id': '1234'}
        values.update(kwargs)
        return AnalysisResult(**values)


class TestDeviceComponent(QiskitTestCase):
    """Test the DeviceComponent class."""

    def test_str(self):
        """Test string representation."""
        q1 = Qubit(1)
        r1 = Resonator(1)
        self.assertEqual('Q1', str(q1))
        self.assertEqual('R1', str(r1))

    def test_to_component(self):
        """Test converting string to component object."""
        q1 = to_component('Q1')
        self.assertIsInstance(q1, Qubit)
        self.assertEqual('Q1', str(q1))
        r1 = to_component('R1')
        self.assertIsInstance(r1, Resonator)
        self.assertEqual('R1', str(r1))
