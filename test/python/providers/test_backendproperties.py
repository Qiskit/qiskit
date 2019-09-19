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

from unittest import SkipTest

class BackendpropertiesTestCase(QiskitTestCase):
    """Test case for Get methods of Backend Properties
    """

    provider_cls = None
    backend_name = 'fake_openpulse_2q'

    def test_gate_error(self):
        """Test getting the gate errors."""
        self.assertEqual(self.backend.properties().gate_error('u1', 0),
                         1.0)
        self.assertEqual(self.backend.properties().gate_error('u1', [0]),
                         1.0)
        self.assertEqual(self.backend.properties().gate_error('cx', [0, 1]),
                         1.0)

    def test_gate_length(self):
        """Test getting the gate duration."""
        self.assertEqual(self.backend.properties().gate_length('u1', 0),
                         0.)
        self.assertEqual(self.backend.properties().gate_length('u3', qubits=[0]),
                         2 * 1.3333 * 1e-9)

    def test_t1(self):
        self.assertEqual(self.backend.properties().test_t1(0),
                         (7.195004210055389e-05, datetime.datetime(2019, 9, 17, 11, 59, 48, 147267)))

    def test_get_gate_property(self):
        self.assertEqual(self.backend.properties().get_gate_property('cx', (0,1), 'gate_error'), 1.0)

    def test_get_qubit_property(self):
        self.assertEqual(self.backend.properties().get_qubit_property(0,'T1'),
                         (7.195004210055389e-05, datetime.datetime(2019, 9, 17, 11, 59, 48, 147267)))
        self.assertEqual(self.backend.properties().get_qubit_property(0,'frequency'),
                         (4919968006.92, datetime.datetime(2019, 9, 17, 11, 59, 48, 147267)))
