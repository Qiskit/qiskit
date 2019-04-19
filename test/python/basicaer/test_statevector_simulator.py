# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""Test StateVectorSimulatorPy."""

import unittest

import numpy as np

from qiskit.providers.basicaer import StatevectorSimulatorPy
from qiskit.test import ReferenceCircuits
from qiskit.test import providers


class StatevectorSimulatorTest(providers.BackendTestCase):
    """Test BasicAer statevector simulator."""

    backend_cls = StatevectorSimulatorPy
    circuit = None

    def test_run_circuit(self):
        """Test final state vector for single circuit run."""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell_no_measure()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1 / 2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1 / 2)

    def test_measure_collapse(self):
        """Test final measurement collapses statevector"""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector(self.circuit)

        # The final state should be EITHER |00> OR |11>
        diff_00 = np.linalg.norm(np.array([1, 0, 0, 0]) - actual)**2
        diff_11 = np.linalg.norm(np.array([0, 0, 0, 1]) - actual)**2
        success = (np.allclose([diff_00, diff_11], [0, 2])
                   or np.allclose([diff_00, diff_11], [2, 0]))
        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
