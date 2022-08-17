# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Fidelity."""

import unittest

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit.algorithms.fidelities import Fidelity
from qiskit.test import QiskitTestCase


@ddt
class TestFidelity(QiskitTestCase):
    """Test Fidelity"""

    def setUp(self):
        super().setUp()
        parameters = ParameterVector("x", 2)

        rx_rotations = QuantumCircuit(2)
        rx_rotations.rx(parameters[0], 0)
        rx_rotations.rx(parameters[1], 1)

        ry_rotations = QuantumCircuit(2)
        ry_rotations.ry(parameters[0], 0)
        ry_rotations.ry(parameters[1], 1)

        plus = QuantumCircuit(2)
        plus.h(0)
        plus.h(1)

        zero = QuantumCircuit(2)

        self._circuit = [rx_rotations, ry_rotations, plus, zero]
        self._sampler = Sampler()
        self._left_params = np.array([[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi, np.pi]])
        self._right_params = np.array([[0, 0], [0, 0], [np.pi / 2, 0], [0, 0]])

    def test_1param_pair(self):
        """test for fidelity with one pair of parameters"""

        fidelity = Fidelity(self._sampler, self._circuit[0], self._circuit[1])
        job = fidelity.run(left_values = self._left_params[0],
                           right_values = self._right_params[0])
        result = job.result()
        np.testing.assert_allclose(result, np.array([1.0]))

    def test_4param_pairs(self):
        """test for fidelity with four pairs of parameters"""

        fidelity = Fidelity(self._sampler, self._circuit[0], self._circuit[1])
        job = fidelity.run(left_values = self._left_params,
                           right_values = self._right_params)
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.25, 0.0]), atol=1e-16)

    def test_symmetry(self):
        """test for fidelity with the same circuit"""

        fidelity = Fidelity(self._sampler, self._circuit[0], self._circuit[0])
        job_1 = fidelity.run(left_values = self._left_params,
                             right_values = self._right_params)
        job_2 = fidelity.run(left_values = self._right_params,
                             right_values = self._left_params)
        results_1 = job_1.result()
        results_2 = job_2.result()
        np.testing.assert_allclose(results_1, results_2, atol=1e-16)

    def test_no_params(self):
        """test for fidelity without parameters"""
        fidelity = Fidelity(self._sampler, self._circuit[2], self._circuit[3])
        job = fidelity.run()
        results = job.result()
        np.testing.assert_allclose(results, np.array([0.25]), atol=1e-16)

    def test_left_param(self):
        """test for fidelity with only left parameters"""
        fidelity = Fidelity(self._sampler, self._circuit[1], self._circuit[3])
        job = fidelity.run(left_values = self._left_params)
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.5, 0.0]), atol=1e-16)

    def test_right_param(self):
        """test for fidelity with only right parameters"""
        fidelity = Fidelity(self._sampler, self._circuit[3], self._circuit[1])
        job = fidelity.run(right_values = self._left_params)
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.5, 0.0]), atol=1e-16)

    # def test_not_set_circuits(self):
    #     """test for fidelity with no circuits during init."""
    #     fidelity = Fidelity(self._sampler)
    #     with self.assertRaises(ValueError):
    #         _ = fidelity.run(left_values = self._left_params,
    #                          right_values = self._right_params)

    def test_set_circuits_during_run(self):
        """test for fidelity with no circuits during init."""
        fidelity = Fidelity(self._sampler)
        job = fidelity.run(left_circuit = [self._circuit[0]],
                           right_circuit = [self._circuit[1]],
                           left_values = self._left_params,
                           right_values = self._right_params)
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.25, 0.0]), atol=1e-16)

    def test_set_single_circuit_during_run(self):
        """test for fidelity with no circuits during init."""
        fidelity = Fidelity(self._sampler)
        # with self.assertRaises(ValueError):
        #     _ = fidelity.run(right_circuit=[self._circuit[1]],
        #                      left_values =self._left_params,
        #                      right_values =self._right_params
        #                      )
        job = fidelity.run(left_circuit=[self._circuit[0]], right_circuit=[self._circuit[1]],
                           left_values =self._left_params,
                           right_values =self._right_params
                           )
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.25, 0.0]), atol=1e-16)

    def test_circuit_list_during_run(self):
        fidelity = Fidelity(self._sampler)
        job = fidelity.run(left_circuit=[self._circuit[0], self._circuit[1], self._circuit[2]],
                           right_circuit=[self._circuit[1]],
                           left_values=self._left_params,
                           right_values=self._right_params
                           )
        results = job.result()
        np.testing.assert_allclose(results, np.array([1.0, 0.5, 0.25, 0.0]), atol=1e-16)

if __name__ == "__main__":
    unittest.main()
