# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.test import QiskitTestCase


class TestComputeUncompute(QiskitTestCase):
    """Test Compute-Uncompute Fidelity class"""

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
        plus.h([0, 1])

        zero = QuantumCircuit(2)

        rx_rotation = QuantumCircuit(2)
        rx_rotation.rx(parameters[0], 0)
        rx_rotation.h(1)

        self._circuit = [rx_rotations, ry_rotations, plus, zero, rx_rotation]
        self._sampler = Sampler()
        self._left_params = np.array([[0, 0], [np.pi / 2, 0], [0, np.pi / 2], [np.pi, np.pi]])
        self._right_params = np.array([[0, 0], [0, 0], [np.pi / 2, 0], [0, 0]])

    def test_1param_pair(self):
        """test for fidelity with one pair of parameters"""
        fidelity = ComputeUncompute(self._sampler)
        job = fidelity.run(
            self._circuit[0], self._circuit[1], self._left_params[0], self._right_params[0]
        )
        result = job.result()
        np.testing.assert_allclose(result.fidelities, np.array([1.0]))

    def test_1param_pair_local(self):
        """test for fidelity with one pair of parameters"""
        fidelity = ComputeUncompute(self._sampler, local=True)
        job = fidelity.run(
            self._circuit[0], self._circuit[1], self._left_params[0], self._right_params[0]
        )
        result = job.result()
        np.testing.assert_allclose(result.fidelities, np.array([1.0]))

    def test_local(self):
        """test difference between local and global fidelity"""
        fidelity_global = ComputeUncompute(self._sampler, local=False)
        fidelity_local = ComputeUncompute(self._sampler, local=True)
        fidelities = []
        for fidelity in [fidelity_global, fidelity_local]:
            job = fidelity.run(self._circuit[2], self._circuit[3])
            result = job.result()
            fidelities.append(result.fidelities[0])
        np.testing.assert_allclose(fidelities, np.array([0.25, 0.5]), atol=1e-16)

    def test_4param_pairs(self):
        """test for fidelity with four pairs of parameters"""
        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        job = fidelity.run(
            [self._circuit[0]] * n, [self._circuit[1]] * n, self._left_params, self._right_params
        )
        results = job.result()
        np.testing.assert_allclose(results.fidelities, np.array([1.0, 0.5, 0.25, 0.0]), atol=1e-16)

    def test_symmetry(self):
        """test for fidelity with the same circuit"""
        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        job_1 = fidelity.run(
            [self._circuit[0]] * n, [self._circuit[0]] * n, self._left_params, self._right_params
        )
        job_2 = fidelity.run(
            [self._circuit[0]] * n, [self._circuit[0]] * n, self._right_params, self._left_params
        )
        results_1 = job_1.result()
        results_2 = job_2.result()
        np.testing.assert_allclose(results_1.fidelities, results_2.fidelities, atol=1e-16)

    def test_no_params(self):
        """test for fidelity without parameters"""
        fidelity = ComputeUncompute(self._sampler)
        job = fidelity.run([self._circuit[2]], [self._circuit[3]])
        results = job.result()
        np.testing.assert_allclose(results.fidelities, np.array([0.25]), atol=1e-16)

        job = fidelity.run([self._circuit[2]], [self._circuit[3]], [], [])
        results = job.result()
        np.testing.assert_allclose(results.fidelities, np.array([0.25]), atol=1e-16)

    def test_left_param(self):
        """test for fidelity with only left parameters"""
        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        job = fidelity.run(
            [self._circuit[1]] * n, [self._circuit[3]] * n, values_1=self._left_params
        )
        results = job.result()
        np.testing.assert_allclose(results.fidelities, np.array([1.0, 0.5, 0.5, 0.0]), atol=1e-16)

    def test_right_param(self):
        """test for fidelity with only right parameters"""
        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        job = fidelity.run(
            [self._circuit[3]] * n, [self._circuit[1]] * n, values_2=self._left_params
        )
        results = job.result()
        np.testing.assert_allclose(results.fidelities, np.array([1.0, 0.5, 0.5, 0.0]), atol=1e-16)

    def test_not_set_circuits(self):
        """test for fidelity with no circuits."""
        fidelity = ComputeUncompute(self._sampler)
        with self.assertRaises(TypeError):
            job = fidelity.run(
                circuits_1=None,
                circuits_2=None,
                values_1=self._left_params,
                values_2=self._right_params,
            )
            job.result()

    def test_circuit_mismatch(self):
        """test for fidelity with different number of left/right circuits."""
        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        with self.assertRaises(ValueError):
            job = fidelity.run(
                [self._circuit[0]] * n,
                [self._circuit[1]] * (n + 1),
                self._left_params,
                self._right_params,
            )
            job.result()

    def test_asymmetric_params(self):
        """test for fidelity when the 2 circuits have different number of
        left/right parameters."""

        fidelity = ComputeUncompute(self._sampler)
        n = len(self._left_params)
        right_params = [[p] for p in self._right_params[:, 0]]
        job = fidelity.run(
            [self._circuit[0]] * n, [self._circuit[4]] * n, self._left_params, right_params
        )
        result = job.result()
        np.testing.assert_allclose(result.fidelities, np.array([0.5, 0.25, 0.25, 0.0]), atol=1e-16)

    def test_input_format(self):
        """test for different input format variations"""

        fidelity = ComputeUncompute(self._sampler)
        circuit = RealAmplitudes(2)
        values = np.random.random(circuit.num_parameters)
        shift = np.ones_like(values) * 0.01

        # lists of circuits, lists of numpy arrays
        job = fidelity.run([circuit], [circuit], [values], [values + shift])
        result_1 = job.result()

        # lists of circuits, lists of lists
        shift_val = values + shift
        job = fidelity.run([circuit], [circuit], [values.tolist()], [shift_val.tolist()])
        result_2 = job.result()

        # circuits, lists
        shift_val = values + shift
        job = fidelity.run(circuit, circuit, values.tolist(), shift_val.tolist())
        result_3 = job.result()

        # circuits, np.arrays
        job = fidelity.run(circuit, circuit, values, values + shift)
        result_4 = job.result()

        np.testing.assert_allclose(result_1.fidelities, result_2.fidelities, atol=1e-16)
        np.testing.assert_allclose(result_1.fidelities, result_3.fidelities, atol=1e-16)
        np.testing.assert_allclose(result_1.fidelities, result_4.fidelities, atol=1e-16)

    def test_input_measurements(self):
        """test for fidelity with measurements on input circuits"""
        fidelity = ComputeUncompute(self._sampler)
        circuit_1 = self._circuit[0]
        circuit_1.measure_all()
        circuit_2 = self._circuit[1]
        circuit_2.measure_all()

        job = fidelity.run(circuit_1, circuit_2, self._left_params[0], self._right_params[0])
        result = job.result()
        np.testing.assert_allclose(result.fidelities, np.array([1.0]))

    def test_options(self):
        """Test fidelity's run options"""
        sampler_shots = Sampler(options={"shots": 1024})

        with self.subTest("sampler"):
            # Only options in sampler
            fidelity = ComputeUncompute(sampler_shots)
            options = fidelity.options
            job = fidelity.run(self._circuit[2], self._circuit[3])
            result = job.result()
            self.assertEqual(options.__dict__, {"shots": 1024})
            self.assertEqual(result.options.__dict__, {"shots": 1024})

        with self.subTest("fidelity init"):
            # Fidelity default options override sampler
            # options and add new fields
            fidelity = ComputeUncompute(sampler_shots, options={"shots": 2048, "dummy": 100})
            options = fidelity.options
            job = fidelity.run(self._circuit[2], self._circuit[3])
            result = job.result()
            self.assertEqual(options.__dict__, {"shots": 2048, "dummy": 100})
            self.assertEqual(result.options.__dict__, {"shots": 2048, "dummy": 100})

        with self.subTest("fidelity update"):
            # Update fidelity options
            fidelity = ComputeUncompute(sampler_shots, options={"shots": 2048, "dummy": 100})
            fidelity.update_default_options(shots=100)
            options = fidelity.options
            job = fidelity.run(self._circuit[2], self._circuit[3])
            result = job.result()
            self.assertEqual(options.__dict__, {"shots": 100, "dummy": 100})
            self.assertEqual(result.options.__dict__, {"shots": 100, "dummy": 100})

        with self.subTest("fidelity run"):
            # Run options override fidelity options
            fidelity = ComputeUncompute(sampler_shots, options={"shots": 2048, "dummy": 100})
            job = fidelity.run(self._circuit[2], self._circuit[3], shots=50, dummy=None)
            options = fidelity.options
            result = job.result()
            # Only default + sampler options. Not run.
            self.assertEqual(options.__dict__, {"shots": 2048, "dummy": 100})
            self.assertEqual(result.options.__dict__, {"shots": 50, "dummy": None})


if __name__ == "__main__":
    unittest.main()
