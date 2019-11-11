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
import warnings

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
        self.assertEqual(self.config.control(0), ControlChannel(0))
