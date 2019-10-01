# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base TestCase for testing Providers."""

import unittest  # pylint: disable=unused-import
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.test.mock import FakeProvider
from qiskit.test import QiskitTestCase
from qiskit.pulse.exceptions import PulseError


class BackendpropertiesTestCase(QiskitTestCase):
    """Test usability methods of backend.properties()."""

    backend = FakeOpenPulse2Q()
    backend_name = 'fake_openpulse_2q'

    def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.properties = self.backend.properties()

    def test_gate_error(self):
        """Test getting the gate errors."""
        self.assertEqual(self.properties.gate_error('u1', 0),
                         self.properties._gates['u1'][(0,)]['gate_error'][0])
        self.assertEqual(self.properties.gate_error('u1', [0]),
                         self.properties._gates['u1'][(0,)]['gate_error'][0])
        self.assertEqual(self.properties.gate_error('cx', [0, 1]),
                         self.properties._gates['cx'][(0, 1)]['gate_error'][0])

        with self.assertRaises(PulseError):
            self.properties.gate_error('u1', 4)

    def test_gate_length(self):
        """Test getting the gate duration."""
        self.assertEqual(self.properties.gate_length('u1', 0),
                         self.properties._gates['u1'][(0,)]['gate_length'][0])
        self.assertEqual(self.properties.gate_length('u3', [0]),
                         self.properties._gates['u3'][(0,)]['gate_length'][0])

    def test_t1(self):
        """Test getting the t1 of given qubit."""
        self.assertEqual(self.properties.t1(0),
                         self.properties._qubits[0]['T1'][0])

    def test_gate_property(self):
        """Test getting the gate properties."""
        self.assertEqual(self.properties.gate_property('cx', (0, 1), 'gate_error'),
                         self.properties._gates['cx'][(0, 1)]['gate_error'])
        self.assertEqual(self.properties.gate_property('cx'),
                         self.properties._gates['cx'])

        with self.assertRaises(PulseError):
            self.properties.gate_property('u1', None, 'gate_error')

    def test_get_qubit_property(self):
        """Test getting the qubit properties."""
        self.assertEqual(self.properties.qubit_property(0, 'T1'),
                         self.properties._qubits[0]['T1'])
        self.assertEqual(self.properties.qubit_property(0, 'frequency'),
                         self.properties._qubits[0]['frequency'])
        self.assertEqual(self.properties.qubit_property(0),
                         self.properties._qubits[0])

    def test_apply_prefix(self):
        """Testing unit conversions."""
        self.assertEqual(self.properties._apply_prefix(71.9500421005539, 'µs'),
                         7.195004210055389e-05)
        self.assertEqual(self.properties._apply_prefix(71.9500421005539, 'ms'),
                         0.0719500421005539)

        with self.assertRaises(PulseError):
            self.properties._apply_prefix(71.9500421005539, 'ws')
