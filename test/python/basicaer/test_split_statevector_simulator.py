# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test SplitStateVectorSimulatorPy."""

import unittest

import numpy as np
import math

from qiskit.providers.basicaer.split_statevector_simulator import SplitStatevectorSimulatorPy
from qiskit.test import ReferenceCircuits
from qiskit.test import providers


class StatevectorSimulatorTest(providers.BackendTestCase):
    """Test BasicAer split_statevector simulator."""

    backend_cls = SplitStatevectorSimulatorPy
    circuit = None

    def test_run_circuit(self):
        """Test that the simulator works for a circuit without measurements in the middle."""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell_no_measure()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector_tree()['value']

        # state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((abs(actual[0]))**2, 1 / 2)
        self.assertEqual(actual[1], 0)
        self.assertEqual(actual[2], 0)
        self.assertAlmostEqual((abs(actual[3]))**2, 1 / 2)

    def test_measure_split(self):
        """Test the result of a single qubit split by one measurement"""
        # Set test circuit
        self.circuit = ReferenceCircuits.h_measure_h()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector_tree()
        initial = actual['value']
        path_0 = actual['path_0']['value']
        path_1 = actual['path_1']['value']

        # initial (before measurement) state is 1/sqrt(2)|00> + 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((initial[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((initial[1]), 1 / math.sqrt(2))
        self.assertEqual(initial[2], 0)
        self.assertEqual(initial[3], 0)
        # path 0 state is 1/sqrt(2)|00> + 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_0[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((path_0[1]), 1 / math.sqrt(2))
        self.assertEqual(path_0[2], 0)
        self.assertEqual(path_0[3], 0)
        # path 1 state is 1/sqrt(2)|00> - 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_1[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((path_1[1]), -1 / math.sqrt(2))
        self.assertEqual(path_1[2], 0)
        self.assertEqual(path_1[3], 0)

    def test_double_measure_split(self):
        """Test the result of a single qubit split by two measurements"""
        # Set test circuit
        self.circuit = ReferenceCircuits.h_measure_h_double()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector_tree()
        initial = actual['value']
        path_0 = actual['path_0']['value']
        path_1 = actual['path_1']['value']
        path_00 = actual['path_0']['path_0']['value']
        path_01 = actual['path_0']['path_1']['value']
        path_10 = actual['path_1']['path_0']['value']
        path_11 = actual['path_1']['path_1']['value']

        prob_0 = actual['path_0_probability']
        prob_1 = actual['path_1_probability']
        prob_00 = actual['path_0']['path_0_probability']
        prob_01 = actual['path_0']['path_1_probability']
        prob_10 = actual['path_1']['path_0_probability']
        prob_11 = actual['path_1']['path_1_probability']

        # initial (before measurement) state is 1/sqrt(2)|00> + 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((initial[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((initial[1]), 1 / math.sqrt(2))
        self.assertAlmostEqual((prob_0), 1 / 2)
        self.assertAlmostEqual((prob_1), 1 / 2)
        self.assertEqual(initial[2], 0)
        self.assertEqual(initial[3], 0)
        # path 0 state is 1/sqrt(2)|00> + 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_0[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((path_0[1]), 1 / math.sqrt(2))
        self.assertAlmostEqual((prob_00), 1 / 2)
        self.assertAlmostEqual((prob_01), 1 / 2)
        self.assertEqual(path_0[2], 0)
        self.assertEqual(path_0[3], 0)
        # path 1 state is 1/sqrt(2)|00> - 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_1[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((path_1[1]), -1 / math.sqrt(2))
        self.assertAlmostEqual((prob_10), 1 / 2)
        self.assertAlmostEqual((prob_11), 1 / 2)
        self.assertEqual(path_1[2], 0)
        self.assertEqual(path_1[3], 0)
        # path 00 state |00> up to a global phase
        self.assertAlmostEqual((path_00[0]), 1)
        self.assertEqual(path_00[1], 0)
        self.assertEqual(path_00[2], 0)
        self.assertEqual(path_00[3], 0)
        # path 01 state |01> up to a global phase
        self.assertAlmostEqual((path_01[1]), 1)
        self.assertEqual(path_01[0], 0)
        self.assertEqual(path_01[2], 0)
        self.assertEqual(path_01[3], 0)
        # path 10 state |00> up to a global phase
        self.assertAlmostEqual((path_10[0]), 1)
        self.assertEqual(path_10[1], 0)
        self.assertEqual(path_10[2], 0)
        self.assertEqual(path_10[3], 0)
        # path 01 state -|01> up to a global phase
        self.assertAlmostEqual((path_11[1]), -1)
        self.assertEqual(path_11[0], 0)
        self.assertEqual(path_11[2], 0)
        self.assertEqual(path_11[3], 0)

    def test_double_measure_split_unequal_probability(self):
        """Test the result of a single qubit split by two measurements,
        when each measurement has unequal probabilities for the possible outcomes"""
        # Set test circuit
        self.circuit = ReferenceCircuits.rx_measure_rx()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector_tree()
        initial = actual['value']
        path_0 = actual['path_0']['value']
        path_1 = actual['path_1']['value']
        path_00 = actual['path_0']['path_0']['value']
        path_01 = actual['path_0']['path_1']['value']
        path_10 = actual['path_1']['path_0']['value']
        path_11 = actual['path_1']['path_1']['value']

        prob_0 = actual['path_0_probability']
        prob_1 = actual['path_1_probability']
        prob_00 = actual['path_0']['path_0_probability']
        prob_01 = actual['path_0']['path_1_probability']
        prob_10 = actual['path_1']['path_0_probability']
        prob_11 = actual['path_1']['path_1_probability']

        # initial (before measurement) state is sqrt(3)/2|00> (- 1/2)j|01>, up to a global phase
        self.assertAlmostEqual((initial[0]), math.sqrt(3) / 2)
        self.assertAlmostEqual((initial[1]), -0.5j)
        self.assertAlmostEqual((prob_0), 3 / 4)
        self.assertAlmostEqual((prob_1), 1 / 4)
        self.assertEqual(initial[2], 0)
        self.assertEqual(initial[3], 0)
        # path 0 state is 1/sqrt(2)|00> + 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_0[0]), math.sqrt(3) / 2)
        self.assertAlmostEqual((path_0[1]), -0.5j)
        self.assertAlmostEqual((prob_00), 3 / 4)
        self.assertAlmostEqual((prob_01), 1 / 4)
        self.assertEqual(path_0[2], 0)
        self.assertEqual(path_0[3], 0)
        # path 1 state is 1/sqrt(2)|00> - 1/sqrt(2)|01>, up to a global phase
        self.assertAlmostEqual((path_1[0]), -0.5)
        self.assertAlmostEqual((path_1[1]), -1j * (math.sqrt(3) / 2))
        self.assertAlmostEqual((prob_10), 1 / 4)
        self.assertAlmostEqual((prob_11), 3 / 4)
        self.assertEqual(path_1[2], 0)
        self.assertEqual(path_1[3], 0)
        # path 00 state is |00> up to a global phase
        self.assertAlmostEqual(abs((path_00[0]**2)), 1)
        self.assertEqual(path_00[1], 0)
        self.assertEqual(path_00[2], 0)
        self.assertEqual(path_00[3], 0)
        # path 01 state is |01> up to a global phase
        self.assertAlmostEqual(abs((path_01[1]**2)), 1)
        self.assertEqual(path_01[0], 0)
        self.assertEqual(path_01[2], 0)
        self.assertEqual(path_01[3], 0)
        # path 10 state is |00> up to a global phase
        self.assertAlmostEqual(abs((path_10[0]**2)), 1)
        self.assertEqual(path_10[1], 0)
        self.assertEqual(path_10[2], 0)
        self.assertEqual(path_10[3], 0)
        # path 01 state is -|01> up to a global phase
        self.assertAlmostEqual(abs((path_11[1]**2)), 1)
        self.assertEqual(path_11[0], 0)
        self.assertEqual(path_11[2], 0)
        self.assertEqual(path_11[3], 0)

    def test_bell_split(self):
        """Test the result of a two qubit bell state split by one measurement"""
        # Set test circuit
        self.circuit = ReferenceCircuits.bell()
        # Execute
        result = super().test_run_circuit()
        actual = result.get_statevector_tree()
        initial = actual['value']
        path_0 = actual['path_0']['value']
        path_1 = actual['path_1']['value']

        prob_0 = actual['path_0_probability']
        prob_1 = actual['path_1_probability']

        # initial (before measurement) state is 1/sqrt(2)|00> + 1/sqrt(2)|11>, up to a global phase
        self.assertAlmostEqual((initial[0]), 1 / math.sqrt(2))
        self.assertAlmostEqual((initial[3]), 1 / math.sqrt(2))
        self.assertAlmostEqual((prob_0), 1 / 2)
        self.assertAlmostEqual((prob_1), 1 / 2)
        self.assertEqual(initial[1], 0)
        self.assertEqual(initial[2], 0)
        # path 0 state is |00>, up to a global phase
        self.assertAlmostEqual((path_0[0]), 1)
        self.assertEqual(path_0[1], 0)
        self.assertEqual(path_0[2], 0)
        self.assertEqual(path_0[3], 0)
        # path 1 state is |11>, up to a global phase
        self.assertAlmostEqual((path_1[3]), 1)
        self.assertEqual(path_1[0], 0)
        self.assertEqual(path_1[1], 0)
        self.assertEqual(path_1[2], 0)


if __name__ == '__main__':
    unittest.main()
