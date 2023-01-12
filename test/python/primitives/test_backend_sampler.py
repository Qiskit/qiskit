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

"""Tests for BackendSampler."""

import unittest
from test import combine

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import BackendSampler, SamplerResult
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.test import QiskitTestCase

BACKENDS = [FakeNairobi(), FakeNairobiV2()]


@ddt
class TestBackendSampler(QiskitTestCase):
    """Test BackendSampler"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1)
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        self._circuit = [hadamard, bell]
        self._target = [
            {0: 0.5, 1: 0.5},
            {0: 0.5, 3: 0.5, 1: 0, 2: 0},
        ]
        self._pqc = RealAmplitudes(num_qubits=2, reps=2)
        self._pqc.measure_all()
        self._pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        self._pqc2.measure_all()
        self._pqc_params = [[0.0] * 6, [1.0] * 6]
        self._pqc_target = [{0: 1}, {0: 0.0148, 1: 0.3449, 2: 0.0531, 3: 0.5872}]
        self._theta = [
            [0, 1, 1, 2, 3, 5],
            [1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]

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
        if not isinstance(prob, list):
            prob = [prob]
        if not isinstance(target, list):
            target = [target]
        self.assertEqual(len(prob), len(target))
        for p, targ in zip(prob, target):
            for key, t_val in targ.items():
                if key in p:
                    self.assertAlmostEqual(p[key], t_val, delta=0.1)
                else:
                    self.assertAlmostEqual(t_val, 0, delta=0.1)

    @combine(backend=BACKENDS)
    def test_sampler_run(self, backend):
        """Test Sampler.run()."""
        bell = self._circuit[1]
        sampler = BackendSampler(backend=backend)
        job = sampler.run(circuits=[bell])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        self.assertIsInstance(result, SamplerResult)
        # print([q.binary_probabilities() for q in result.quasi_dists])
        self._compare_probs(result.quasi_dists, self._target[1])

    @combine(backend=BACKENDS)
    def test_sample_run_multiple_circuits(self, backend):
        """Test Sampler.run() with multiple circuits."""
        # executes three Bell circuits
        # Argument `parameters` is optional.
        bell = self._circuit[1]
        sampler = BackendSampler(backend=backend)
        result = sampler.run([bell, bell, bell]).result()
        # print([q.binary_probabilities() for q in result.quasi_dists])
        self._compare_probs(result.quasi_dists[0], self._target[1])
        self._compare_probs(result.quasi_dists[1], self._target[1])
        self._compare_probs(result.quasi_dists[2], self._target[1])

    @combine(backend=BACKENDS)
    def test_sampler_run_with_parameterized_circuits(self, backend):
        """Test Sampler.run() with parameterized circuits."""
        # parameterized circuit

        pqc = self._pqc
        pqc2 = self._pqc2
        theta1, theta2, theta3 = self._theta

        sampler = BackendSampler(backend=backend)
        result = sampler.run([pqc, pqc, pqc2], [theta1, theta2, theta3]).result()

        # result of pqc(theta1)
        prob1 = {
            "00": 0.1309248462975777,
            "01": 0.3608720796028448,
            "10": 0.09324865232050054,
            "11": 0.41495442177907715,
        }
        self.assertDictAlmostEqual(result.quasi_dists[0].binary_probabilities(), prob1, delta=0.1)

        # result of pqc(theta2)
        prob2 = {
            "00": 0.06282290651933871,
            "01": 0.02877144385576705,
            "10": 0.606654494132085,
            "11": 0.3017511554928094,
        }
        self.assertDictAlmostEqual(result.quasi_dists[1].binary_probabilities(), prob2, delta=0.1)

        # result of pqc2(theta3)
        prob3 = {
            "00": 0.1880263994380416,
            "01": 0.6881971261189544,
            "10": 0.09326232720582443,
            "11": 0.030514147237179892,
        }
        self.assertDictAlmostEqual(result.quasi_dists[2].binary_probabilities(), prob3, delta=0.1)

    @combine(backend=BACKENDS)
    def test_run_1qubit(self, backend):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        sampler = BackendSampler(backend=backend)
        result = sampler.run([qc, qc2]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, 0.1)

    @combine(backend=BACKENDS)
    def test_run_2qubit(self, backend):
        """test for 2-qubit cases"""
        qc0 = QuantumCircuit(2)
        qc0.measure_all()
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.measure_all()
        qc3 = QuantumCircuit(2)
        qc3.x([0, 1])
        qc3.measure_all()

        sampler = BackendSampler(backend=backend)
        result = sampler.run([qc0, qc1, qc2, qc3]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[2], {2: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[3], {3: 1}, 0.1)

    @combine(backend=BACKENDS)
    def test_run_errors(self, backend):
        """Test for errors"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        sampler = BackendSampler(backend=backend)
        with self.assertRaises(ValueError):
            sampler.run([qc1], [[1e2]]).result()
        with self.assertRaises(ValueError):
            sampler.run([qc2], [[]]).result()
        with self.assertRaises(ValueError):
            sampler.run([qc2], [[1e2]]).result()

    @combine(backend=BACKENDS)
    def test_run_empty_parameter(self, backend):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        sampler = BackendSampler(backend=backend)
        with self.subTest("one circuit"):
            result = sampler.run([qc], shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 1)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictAlmostEqual(quasi_dist, {0: 1.0}, delta=0.1)
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            result = sampler.run([qc, qc], shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 2)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictAlmostEqual(quasi_dist, {0: 1.0}, delta=0.1)
            self.assertEqual(len(result.metadata), 2)

    @combine(backend=BACKENDS)
    def test_run_numpy_params(self, backend):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        qc.measure_all()
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        sampler = BackendSampler(backend=backend)
        target = sampler.run([qc] * k, params_list).result()

        with self.subTest("ndarrary"):
            result = sampler.run([qc] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictAlmostEqual(result.quasi_dists[i], target.quasi_dists[i], delta=0.1)

        with self.subTest("list of ndarray"):
            result = sampler.run([qc] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictAlmostEqual(result.quasi_dists[i], target.quasi_dists[i], delta=0.1)

    @combine(backend=BACKENDS)
    def test_run_with_shots_option(self, backend):
        """test with shots option."""
        params, target = self._generate_params_target([1])
        sampler = BackendSampler(backend=backend)
        result = sampler.run(
            circuits=[self._pqc], parameter_values=params, shots=1024, seed=15
        ).result()
        self._compare_probs(result.quasi_dists, target)

    @combine(backend=BACKENDS)
    def test_primitive_job_status_done(self, backend):
        """test primitive job's status"""
        bell = self._circuit[1]
        sampler = BackendSampler(backend=backend)
        job = sampler.run(circuits=[bell])
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_primitive_job_size_limit_backend_v2(self):
        """Test primitive respects backend's job size limit."""

        class FakeNairobiLimitedCircuits(FakeNairobiV2):
            """FakeNairobiV2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = BackendSampler(backend=FakeNairobiLimitedCircuits())
        result = sampler.run([qc, qc2]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, 0.1)

    def test_primitive_job_size_limit_backend_v1(self):
        """Test primitive respects backend's job size limit."""
        backend = FakeNairobi()
        config = backend.configuration()
        config.max_experiments = 1
        backend._configuration = config
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = BackendSampler(backend=backend)
        result = sampler.run([qc, qc2]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, 0.1)
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, 0.1)

    def test_sequential_run(self):
        """Test sequential run."""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()
        sampler = BackendSampler(backend=FakeNairobi())
        result = sampler.run([qc]).result()
        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, 0.1)
        result2 = sampler.run([qc2]).result()
        self.assertDictAlmostEqual(result2.quasi_dists[0], {1: 1}, 0.1)
        result3 = sampler.run([qc, qc2]).result()
        self.assertDictAlmostEqual(result3.quasi_dists[0], {0: 1}, 0.1)
        self.assertDictAlmostEqual(result3.quasi_dists[1], {1: 1}, 0.1)


if __name__ == "__main__":
    unittest.main()
