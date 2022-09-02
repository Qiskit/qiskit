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
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import BackendSampler, SamplerResult
from qiskit.providers import JobStatus, JobV1
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2

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

    @combine(indices=[[0], [1], [0, 1]], backend=BACKENDS)
    def test_sampler(self, indices, backend):
        """test for sampler"""
        circuits, target = self._generate_circuits_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=circuits)
            result = sampler(list(range(len(indices))), parameter_values=[[] for _ in indices])
        self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0], [1], [0, 1]], backend=BACKENDS)
    def test_sampler_pqc(self, indices, backend):
        """test for sampler with a parametrized circuit"""
        params, target = self._generate_params_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=self._pqc)
            result = sampler([0] * len(params), params)
        self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0, 0], [0, 1], [1, 1]], backend=BACKENDS)
    def test_evaluate_two_pqcs(self, indices, backend):
        """test for sampler with two parametrized circuits"""
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=circs)
            result = sampler(indices, parameter_values=params)
        self._compare_probs(result.quasi_dists, target)

    @combine(backend=BACKENDS)
    def test_sampler_example(self, backend):
        """test for Sampler example"""

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        # executes a Bell circuit
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=[bell], parameters=[[]])
            result = sampler(parameter_values=[[]], circuits=[0])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 1)
        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 0.5, 3: 0.5}, delta=0.1)
        self.assertEqual(len(result.metadata), 1)

        # executes three Bell circuits
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=[bell] * 3, parameters=[[]] * 3)
            result = sampler([0, 1, 2], [[]] * 3)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)
        for dist in result.quasi_dists:
            self.assertDictAlmostEqual(dist, {0: 0.5, 3: 0.5}, delta=0.1)

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=[bell])
            result = sampler([bell, bell, bell])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)
        for dist in result.quasi_dists:
            self.assertDictAlmostEqual(dist, {0: 0.5, 3: 0.5}, delta=0.1)

        # parametrized circuit
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [1, 2, 3, 4, 5, 6]
        theta3 = [0, 1, 2, 3, 4, 5, 6, 7]

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend, circuits=[pqc, pqc2], parameters=[pqc.parameters, pqc2.parameters]
            )
            result = sampler([0, 0, 1], [theta1, theta2, theta3])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(
            values,
            [0.13092484629757767, 0.3608720796028449, 0.09324865232050054, 0.414954421779077],
            atol=0.1,
        )

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(
            values,
            [0.06282290651933871, 0.02877144385576703, 0.606654494132085, 0.3017511554928095],
            atol=0.1,
        )

        keys, values = zip(*sorted(result.quasi_dists[2].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(
            values,
            [
                0.18802639943804164,
                0.6881971261189544,
                0.09326232720582446,
                0.030514147237179882,
            ],
            atol=0.1,
        )

    @combine(backend=BACKENDS)
    def test_sampler_param_order(self, backend):
        """test for sampler with different parameter orders"""
        x = Parameter("x")
        y = Parameter("y")

        qc = QuantumCircuit(3, 3)
        qc.rx(x, 0)
        qc.rx(y, 1)
        qc.x(2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend, circuits=[qc, qc], parameters=[[x, y], [y, x]]
            )
            result = sampler([0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[0], {4: 1}, delta=0.1)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[1], {4: 1}, delta=0.1)

        # qc({x: pi/2, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[2], {4: 0.5, 5: 0.5}, delta=0.1)

        # qc({x: 0, y: pi/2})
        self.assertDictAlmostEqual(result.quasi_dists[3], {4: 0.5, 6: 0.5}, delta=0.1)

    @combine(backend=BACKENDS)
    def test_sampler_reverse_meas_order(self, backend):
        """test for sampler with reverse measurement order"""
        x = Parameter("x")
        y = Parameter("y")

        qc = QuantumCircuit(3, 3)
        qc.rx(x, 0)
        qc.rx(y, 1)
        qc.x(2)
        qc.measure(0, 2)
        qc.measure(1, 1)
        qc.measure(2, 0)

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend, circuits=[qc, qc], parameters=[[x, y], [y, x]]
            )
            result = sampler([0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[0], {1: 1}, delta=0.1)

        # qc({x: 0, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, delta=0.1)

        # qc({x: pi/2, y: 0})
        self.assertDictAlmostEqual(result.quasi_dists[2], {1: 0.5, 5: 0.5}, delta=0.1)

        # qc({x: 0, y: pi/2})
        self.assertDictAlmostEqual(result.quasi_dists[3], {1: 0.5, 3: 0.5}, delta=0.1)

    @combine(backend=BACKENDS)
    def test_1qubit(self, backend):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend, circuits=[qc, qc2], parameters=[qc.parameters, qc2.parameters]
            )
            result = sampler([0, 1], [[]] * 2)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, (0,))
        np.testing.assert_allclose(values, [1])

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, (1,))
        np.testing.assert_allclose(values, [1])

    @combine(backend=BACKENDS)
    def test_2qubit(self, backend):
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

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend,
                circuits=[qc0, qc1, qc2, qc3],
                parameters=[qc0.parameters, qc1.parameters, qc2.parameters, qc3.parameters],
            )
            result = sampler([0, 1, 2, 3], [[]] * 4)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        self.assertDictAlmostEqual(result.quasi_dists[0], {0: 1}, delta=0.1)
        self.assertDictAlmostEqual(result.quasi_dists[1], {1: 1}, delta=0.1)
        self.assertDictAlmostEqual(result.quasi_dists[2], {2: 1}, delta=0.1)
        self.assertDictAlmostEqual(result.quasi_dists[3], {3: 1}, delta=0.1)

    @combine(backend=BACKENDS)
    def test_errors(self, backend):
        """Test for errors"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(
                backend=backend, circuits=[qc1, qc2], parameters=[qc1.parameters, qc2.parameters]
            )
        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            sampler([0], [[1e2]])
        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            sampler([1], [[]])
        with self.assertRaises(QiskitError), self.assertWarns(DeprecationWarning):
            sampler([1], [[1e2]])

    @combine(backend=BACKENDS)
    def test_empty_parameter(self, backend):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        with self.assertWarns(DeprecationWarning):
            sampler = BackendSampler(backend=backend, circuits=[qc] * 10)
        with self.subTest("one circuit"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([0], shots=1000)
            self.assertEqual(len(result.quasi_dists), 1)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictAlmostEqual(quasi_dist, {0: 1.0}, delta=0.1)
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([2, 4], shots=1000)
            self.assertEqual(len(result.quasi_dists), 2)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictAlmostEqual(quasi_dist, {0: 1.0}, delta=0.1)
            self.assertEqual(len(result.metadata), 2)

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
        with self.assertRaises(QiskitError):
            sampler.run([qc1], [[1e2]]).result()
        with self.assertRaises(QiskitError):
            sampler.run([qc2], [[]]).result()
        with self.assertRaises(QiskitError):
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


if __name__ == "__main__":
    unittest.main()
