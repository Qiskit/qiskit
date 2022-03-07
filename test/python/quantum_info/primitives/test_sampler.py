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

"""Tests for Sampler."""

from test import combine

from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Sampler
from qiskit.test import QiskitTestCase


@ddt
class TestSampler(QiskitTestCase):
    """Test Sampler"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1)
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure(0, 0)
        bell.measure(1, 1)
        self._circuit = [hadamard, bell]
        self._target = [
            {0: 0.5, 1: 0.5},
            {0: 0.5, 3: 0.5, 1: 0, 2: 0},
        ]
        self._pqc = QuantumCircuit(2, 2)
        self._pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
        self._pqc.measure(0, 0)
        self._pqc.measure(1, 1)
        self._pqc_params = [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self._pqc_target = [{0: 1}, {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}]

    def _generate_circuits_target(self, indices):
        if isinstance(indices, list):
            circuits = [self._circuit[j] for j in indices]
            target = [self._target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return circuits, target

    def _generate_params_target(self, indices):
        if isinstance(indices, int):
            params = self._pqc_params[indices]
            target = self._pqc_target[indices]
        elif isinstance(indices, list):
            params = [self._pqc_params[j] for j in indices]
            target = [self._pqc_target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return params, target

    def _compare_probs(self, prob, target):
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(prob), len(target))
        for p, targ in zip(prob, target):
            for key, t_val in targ.items():
                if key in p:
                    self.assertAlmostEqual(p[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @combine(indices=[[0], [1], [0, 1]])
    def test_evaluate(self, indices):
        """test for evaluate"""
        circuits, target = self._generate_circuits_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circuits) as sampler:
                result = sampler(parameters=[[] for _ in indices])
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circuits)
            result = sampler(parameters=[[] for _ in indices])
            self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0], [1], [0, 1]])
    def test_evaluate_pqc(self, indices):
        """test for evaluate a parametrized circuit"""
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc) as sampler:
                result = sampler(parameters=params)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc)
            result = sampler(parameters=params)
            self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0, 0], [0, 1], [1, 1]])
    def test_evaluate_two_pqcs(self, indices):
        """test for evaluate two parametrized circuits"""
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circs) as sampler:
                result = sampler(parameters=params)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circs)
            result = sampler(parameters=params)
            self._compare_probs(result.quasi_dists, target)
