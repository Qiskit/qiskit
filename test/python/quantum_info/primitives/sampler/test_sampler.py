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

import unittest
from test import combine

from ddt import ddt

from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info.primitives.results import SamplerResult
from qiskit.quantum_info.primitives.sampler import Sampler
from qiskit.test import QiskitTestCase
from qiskit.utils import has_aer

if has_aer():
    from qiskit import Aer


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
        self._run_config = {"seed_simulator": 15}
        self._pqc = QuantumCircuit(2, 2)
        self._pqc.compose(RealAmplitudes(num_qubits=2, reps=2), inplace=True)
        self._pqc.measure(0, 0)
        self._pqc.measure(1, 1)

    def _generate_circuits_target(self, indices):
        if isinstance(indices, int):
            circuits = self._circuit[indices]
            target = self._target[indices]
        elif isinstance(indices, list):
            circuits = [self._circuit[j] for j in indices]
            target = [self._target[j] for j in indices]
        else:
            raise ValueError(f"invalid index {indices}")
        return circuits, target

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

    @combine(indices=[0, 1, [0, 1]], shots=[1000, 2000])
    def test_evaluate_basicaer(self, indices, shots):
        """test for evaluate"""
        backend = BasicAer.get_backend("qasm_simulator")
        circuits, target = self._generate_circuits_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circuits, backend=backend) as sampler:
                result: SamplerResult = sampler(shots=shots, **self._run_config)
                self.assertEqual(result.shots, shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circuits, backend=backend)
            result = sampler(shots=shots, **self._run_config)
            self.assertEqual(result.shots, shots)
            self._compare_probs(result.quasi_dists, target)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    @combine(indices=[0, 1, [0, 1]], shots=[1000, 2000])
    def test_evaluate_aer(self, indices, shots):
        """test for evaluate"""
        backend = Aer.get_backend("aer_simulator")
        circuits, target = self._generate_circuits_target(indices)
        with self.subTest("with-guard"):
            with Sampler(circuits=circuits, backend=backend) as sampler:
                result: SamplerResult = sampler(shots=shots, **self._run_config)
                self.assertEqual(result.shots, shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=circuits, backend=backend)
            result = sampler(shots=shots, **self._run_config)
            self.assertEqual(result.shots, shots)
            self._compare_probs(result.quasi_dists, target)

    @combine(
        params_target=[
            ([0, 0, 0, 0, 0, 0], {0: 1}),
            ([1, 1, 1, 1, 1, 1], {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}),
        ],
        shots=[1000, 2000],
    )
    def test_evaluate_pqc_basicaer(self, params_target, shots):
        """test for evaluate parametrized circuits"""
        backend = BasicAer.get_backend("qasm_simulator")
        params, target = params_target
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(params)
                self.assertEqual(result.shots, shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(params)
            self.assertEqual(result.shots, shots)
            self._compare_probs(result.quasi_dists, target)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    @combine(
        params_target=[
            ([0, 0, 0, 0, 0, 0], {0: 1}),
            ([1, 1, 1, 1, 1, 1], {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}),
        ],
        shots=[1000, 2000],
    )
    def test_evaluate_pqc_aer(self, params_target, shots):
        """test for evaluate parametrized circuits"""
        backend = Aer.get_backend("aer_simulator")
        params, target = params_target
        with self.subTest("with-guard"):
            with Sampler(circuits=self._pqc, backend=backend) as sampler:
                sampler.set_run_options(shots=shots, **self._run_config)
                result: SamplerResult = sampler(params)
                self.assertEqual(result.shots, shots)
                self._compare_probs(result.quasi_dists, target)

        with self.subTest("direct call"):
            sampler = Sampler(circuits=self._pqc, backend=backend)
            sampler.set_run_options(shots=shots, **self._run_config)
            result = sampler(params)
            self.assertEqual(result.shots, shots)
            self._compare_probs(result.quasi_dists, target)
