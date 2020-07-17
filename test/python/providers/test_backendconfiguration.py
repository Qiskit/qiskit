# -*- coding: utf-8 -*-

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
"""
Test that the PulseBackendConfiguration methods work as expected with a mocked Pulse backend.
"""
import collections
import copy

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider

from qiskit.pulse.channels import DriveChannel, MeasureChannel, ControlChannel, AcquireChannel
from qiskit.providers import BackendConfigurationError


class TestBackendConfiguration(QiskitTestCase):
    """Test the methods on the BackendConfiguration class."""

    def setUp(self):
        self.provider = FakeProvider()
        self.config = self.provider.get_backend('fake_openpulse_2q').configuration()

    def test_simple_config(self):
        """Test the most basic getters."""
        self.assertEqual(self.config.dt, 1.3333 * 1.e-9)
        self.assertEqual(self.config.dtm, 10.5 * 1.e-9)
        self.assertEqual(self.config.basis_gates, ['u1', 'u2', 'u3', 'cx', 'id'])

    def test_sample_rate(self):
        """Test that sample rate is 1/dt."""
        self.assertEqual(self.config.sample_rate, 1. / self.config.dt)

    def test_hamiltonian(self):
        """Test the hamiltonian method."""
        self.assertEqual(self.config.hamiltonian['description'],
                         "A hamiltonian for a mocked 2Q device, with 1Q and 2Q terms.")
        # 3Q doesn't offer a hamiltonian -- test that we get a reasonable response
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertEqual(backend_3q.configuration().hamiltonian, None)

    def test_get_channels(self):
        """Test requesting channels from the system."""
        self.assertEqual(self.config.drive(0), DriveChannel(0))
        self.assertEqual(self.config.measure(1), MeasureChannel(1))
        self.assertEqual(self.config.acquire(0), AcquireChannel(0))
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if the system doesn't have that many qubits
            self.assertEqual(self.config.acquire(10), AcquireChannel(10))
        self.assertEqual(self.config.control(qubits=[0, 1]), [ControlChannel(0)])
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if key not found in self._qubit_channel_map
            self.config.control(qubits=(10, 1))

    def test_get_channel_qubits(self):
        """Test to get all qubits operated on a given channel."""
        self.assertEqual(self.config.get_channel_qubits(channel=DriveChannel(0)), [0])
        self.assertEqual(self.config.get_channel_qubits(channel=ControlChannel(0)), [0, 1])
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertEqual(backend_3q.configuration().get_channel_qubits(ControlChannel(2)), [2, 1])
        self.assertEqual(backend_3q.configuration().get_channel_qubits(ControlChannel(1)), [1, 0])
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if key not found in self._channel_qubit_map
            self.config.get_channel_qubits(MeasureChannel(10))

    def test_get_qubit_channels(self):
        """Test to get all channels operated on a given qubit."""
        self.assertTrue(self._test_lists_equal(
            actual=self.config.get_qubit_channels(qubit=(1,)),
            expected=[DriveChannel(1), MeasureChannel(1), AcquireChannel(1)]
        ))
        self.assertTrue(self._test_lists_equal(
            actual=self.config.get_qubit_channels(qubit=1),
            expected=[ControlChannel(0), ControlChannel(1), AcquireChannel(1),
                      DriveChannel(1), MeasureChannel(1)]
        ))
        backend_3q = self.provider.get_backend('fake_openpulse_3q')
        self.assertTrue(self._test_lists_equal(
            actual=backend_3q.configuration().get_qubit_channels(1),
            expected=[MeasureChannel(1), ControlChannel(0), ControlChannel(2),
                      AcquireChannel(1), DriveChannel(1), ControlChannel(1)]
        ))
        with self.assertRaises(BackendConfigurationError):
            # Check that an error is raised if key not found in self._channel_qubit_map
            self.config.get_qubit_channels(10)

    def test_get_rep_times(self):
        """Test whether rep time property is the right size"""
        _rep_times_us = [100, 250, 500, 1000]
        _rep_times_s = [_rt * 1.e-6 for _rt in _rep_times_us]

        for i, time in enumerate(_rep_times_s):
            self.assertAlmostEqual(self.config.rep_times[i], time)
        for i, time in enumerate(_rep_times_us):
            self.assertEqual(round(self.config.rep_times[i]*1e6), time)
        for rep_time in self.config.to_dict()['rep_times']:
            self.assertGreater(rep_time, 0)

    def test_get_channel_prefix_index(self):
        """Test private method to get channel and index."""
        self.assertEqual(self.config._get_channel_prefix_index('acquire0'), ('acquire', 0))
        with self.assertRaises(BackendConfigurationError):
            self.config._get_channel_prefix_index("acquire")

    def _test_lists_equal(self, actual, expected):
        """Test if 2 lists are equal. It returns ``True`` is lists are equal."""
        return collections.Counter(actual) == collections.Counter(expected)

    def test_deepcopy(self):
        """Ensure that a deepcopy succeeds and results in an identical object."""
        copy_config = copy.deepcopy(self.config)
        print(copy_config.to_dict())
        print("Original:")
        print(self.config.to_dict())
        self.assertEqual(copy_config, self.config)
