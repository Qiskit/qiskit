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

"""Tests for Sampler."""

import unittest
from test import combine

import numpy as np
from ddt import ddt

from qiskit import QuantumCircuit, pulse, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.exceptions import QiskitError
from qiskit.primitives import Sampler, SamplerResult
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.fake_provider import FakeAlmaden
from qiskit.test import QiskitTestCase


@ddt
class TestSampler(QiskitTestCase):
    """Test Sampler"""

    def setUp(self):
        super().setUp()
        hadamard = QuantumCircuit(1, 1, name="Hadamard")
        hadamard.h(0)
        hadamard.measure(0, 0)
        bell = QuantumCircuit(2, name="Bell")
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
                    self.assertAlmostEqual(p[key], t_val, places=1)
                else:
                    self.assertAlmostEqual(t_val, 0, places=1)

    @combine(indices=[[0], [1], [0, 1]])
    def test_sampler(self, indices):
        """test for sampler"""
        circuits, target = self._generate_circuits_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=circuits)
            result = sampler(list(range(len(indices))), parameter_values=[[] for _ in indices])
        self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0], [1], [0, 1]])
    def test_sampler_pqc(self, indices):
        """test for sampler with a parametrized circuit"""
        params, target = self._generate_params_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=self._pqc)
            result = sampler([0] * len(params), params)
        self._compare_probs(result.quasi_dists, target)

    @combine(indices=[[0, 0], [0, 1], [1, 1]])
    def test_evaluate_two_pqcs(self, indices):
        """test for sampler with two parametrized circuits"""
        circs = [self._pqc, self._pqc]
        params, target = self._generate_params_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=circs)
            result = sampler(indices, parameter_values=params)
        self._compare_probs(result.quasi_dists, target)

    def test_sampler_example(self):
        """test for Sampler example"""

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()

        # executes a Bell circuit
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=[bell], parameters=[[]])
            result = sampler(parameter_values=[[]], circuits=[0])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 1)
        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0.5, 0, 0, 0.5])
        self.assertEqual(len(result.metadata), 1)

        # executes three Bell circuits
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler([bell] * 3, [[]] * 3)
            result = sampler([0, 1, 2], [[]] * 3)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)
        for dist in result.quasi_dists:
            keys, values = zip(*sorted(dist.items()))
            self.assertTupleEqual(keys, tuple(range(4)))
            np.testing.assert_allclose(values, [0.5, 0, 0, 0.5])

        with self.assertWarns(DeprecationWarning):
            sampler = Sampler([bell])
            result = sampler([bell, bell, bell])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)
        for dist in result.quasi_dists:
            keys, values = zip(*sorted(dist.items()))
            self.assertTupleEqual(keys, tuple(range(4)))
            np.testing.assert_allclose(values, [0.5, 0, 0, 0.5])

        # parametrized circuit
        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [1, 2, 3, 4, 5, 6]
        theta3 = [0, 1, 2, 3, 4, 5, 6, 7]

        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=[pqc, pqc2], parameters=[pqc.parameters, pqc2.parameters])
            result = sampler([0, 0, 1], [theta1, theta2, theta3])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 3)
        self.assertEqual(len(result.metadata), 3)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(
            values,
            [0.13092484629757767, 0.3608720796028449, 0.09324865232050054, 0.414954421779077],
        )

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(
            values,
            [0.06282290651933871, 0.02877144385576703, 0.606654494132085, 0.3017511554928095],
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
        )

    def test_sampler_param_order(self):
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
            sampler = Sampler([qc, qc], [[x, y], [y, x]])
            result = sampler([0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0, 0, 0, 1, 0, 0, 0])

        # qc({x: 0, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0, 0, 0, 1, 0, 0, 0])

        # qc({x: pi/2, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[2].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0, 0, 0, 0.5, 0.5, 0, 0])

        # qc({x: 0, y: pi/2})
        keys, values = zip(*sorted(result.quasi_dists[3].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0, 0, 0, 0.5, 0, 0.5, 0])

    def test_sampler_reverse_meas_order(self):
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
            sampler = Sampler([qc, qc], [[x, y], [y, x]])
            result = sampler([0, 1, 0, 1], [[0, 0], [0, 0], [np.pi / 2, 0], [np.pi / 2, 0]])
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        # qc({x: 0, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 1, 0, 0, 0, 0, 0, 0])

        # qc({x: 0, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 1, 0, 0, 0, 0, 0, 0])

        # qc({x: pi/2, y: 0})
        keys, values = zip(*sorted(result.quasi_dists[2].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0.5, 0, 0, 0, 0.5, 0, 0])

        # qc({x: 0, y: pi/2})
        keys, values = zip(*sorted(result.quasi_dists[3].items()))
        self.assertTupleEqual(keys, tuple(range(8)))
        np.testing.assert_allclose(values, [0, 0.5, 0, 0.5, 0, 0, 0, 0])

    def test_1qubit(self):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        with self.assertWarns(DeprecationWarning):
            sampler = Sampler([qc, qc2], [qc.parameters, qc2.parameters])
            result = sampler([0, 1], [[]] * 2)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(2)))
        np.testing.assert_allclose(values, [1, 0])

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(2)))
        np.testing.assert_allclose(values, [0, 1])

    def test_2qubit(self):
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
            sampler = Sampler(
                [qc0, qc1, qc2, qc3],
                [qc0.parameters, qc1.parameters, qc2.parameters, qc3.parameters],
            )
            result = sampler([0, 1, 2, 3], [[]] * 4)
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [1, 0, 0, 0])

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 1, 0, 0])

        keys, values = zip(*sorted(result.quasi_dists[2].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 0, 1, 0])

        keys, values = zip(*sorted(result.quasi_dists[3].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 0, 0, 1])

    def test_errors(self):
        """Test for errors"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()

        with self.assertWarns(DeprecationWarning):
            sampler = Sampler([qc1, qc2], [qc1.parameters, qc2.parameters])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            sampler([0], [[1e2]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            sampler([1], [[]])
        with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
            sampler([1], [[1e2]])

    def test_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=[qc] * 10)
        with self.subTest("one circuit"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([0], shots=1000)
            self.assertEqual(len(result.quasi_dists), 1)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([2, 4], shots=1000)
            self.assertEqual(len(result.quasi_dists), 2)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 2)

    def test_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        qc.measure_all()
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=qc)
            target = sampler([0] * k, params_list)

        with self.subTest("ndarrary"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([0] * k, params_array)
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictEqual(result.quasi_dists[i], target.quasi_dists[i])

        with self.subTest("list of ndarray"):
            with self.assertWarns(DeprecationWarning):
                result = sampler([0] * k, params_list_array)
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictEqual(result.quasi_dists[i], target.quasi_dists[i])

    def test_passing_objects(self):
        """Test passing objects for Sampler."""

        params, target = self._generate_params_target([0])

        with self.subTest("Valid test"):
            with self.assertWarns(DeprecationWarning):
                sampler = Sampler(circuits=self._pqc)
                result = sampler(circuits=[self._pqc], parameter_values=params)
            self._compare_probs(result.quasi_dists, target)

        with self.subTest("Invalid circuit test"):
            circuit = QuantumCircuit(2)
            with self.assertWarns(DeprecationWarning):
                sampler = Sampler(circuits=self._pqc)
            with self.assertRaises(ValueError), self.assertWarns(DeprecationWarning):
                result = sampler(circuits=[circuit], parameter_values=params)

    @combine(indices=[[0], [1], [0, 1]])
    def test_deprecated_circuit_indices(self, indices):
        """Test for deprecated arguments"""
        circuits, target = self._generate_circuits_target(indices)
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=circuits)
            result = sampler(
                circuit_indices=list(range(len(indices))),
                parameter_values=[[] for _ in indices],
            )
        self._compare_probs(result.quasi_dists, target)

    def test_with_shots_option(self):
        """test with shots option."""
        params, target = self._generate_params_target([1])
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler(circuits=self._pqc)
            result = sampler(circuits=[0], parameter_values=params, shots=1024, seed=15)
        self._compare_probs(result.quasi_dists, target)
        self.assertEqual(result.quasi_dists[0].shots, 1024)

    def test_with_shots_option_none(self):
        """test with shots=None option. Seed is ignored then."""
        with self.assertWarns(DeprecationWarning):
            sampler = Sampler([self._pqc])
            result_42 = sampler([0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=42)
            result_15 = sampler([0], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=15)
        self.assertDictAlmostEqual(result_42.quasi_dists, result_15.quasi_dists)

    def test_sampler_run(self):
        """Test Sampler.run()."""
        bell = self._circuit[1]
        sampler = Sampler()
        job = sampler.run(circuits=[bell])
        self.assertIsInstance(job, JobV1)
        result = job.result()
        self.assertIsInstance(result, SamplerResult)
        # print([q.binary_probabilities() for q in result.quasi_dists])
        self._compare_probs(result.quasi_dists, self._target[1])

    def test_sample_run_multiple_circuits(self):
        """Test Sampler.run() with multiple circuits."""
        # executes three Bell circuits
        # Argument `parameters` is optional.
        bell = self._circuit[1]
        sampler = Sampler()
        result = sampler.run([bell, bell, bell]).result()
        # print([q.binary_probabilities() for q in result.quasi_dists])
        self._compare_probs(result.quasi_dists[0], self._target[1])
        self._compare_probs(result.quasi_dists[1], self._target[1])
        self._compare_probs(result.quasi_dists[2], self._target[1])

    def test_sampler_run_with_parameterized_circuits(self):
        """Test Sampler.run() with parameterized circuits."""
        # parameterized circuit

        pqc = self._pqc
        pqc2 = self._pqc2
        theta1, theta2, theta3 = self._theta

        sampler = Sampler()
        result = sampler.run([pqc, pqc, pqc2], [theta1, theta2, theta3]).result()

        # result of pqc(theta1)
        prob1 = {
            "00": 0.1309248462975777,
            "01": 0.3608720796028448,
            "10": 0.09324865232050054,
            "11": 0.41495442177907715,
        }
        self.assertDictAlmostEqual(result.quasi_dists[0].binary_probabilities(), prob1)

        # result of pqc(theta2)
        prob2 = {
            "00": 0.06282290651933871,
            "01": 0.02877144385576705,
            "10": 0.606654494132085,
            "11": 0.3017511554928094,
        }
        self.assertDictAlmostEqual(result.quasi_dists[1].binary_probabilities(), prob2)

        # result of pqc2(theta3)
        prob3 = {
            "00": 0.1880263994380416,
            "01": 0.6881971261189544,
            "10": 0.09326232720582443,
            "11": 0.030514147237179892,
        }
        self.assertDictAlmostEqual(result.quasi_dists[2].binary_probabilities(), prob3)

    def test_run_1qubit(self):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        sampler = Sampler()
        result = sampler.run([qc, qc2]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 2)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(2)))
        np.testing.assert_allclose(values, [1, 0])

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(2)))
        np.testing.assert_allclose(values, [0, 1])

    def test_run_2qubit(self):
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

        sampler = Sampler()
        result = sampler.run([qc0, qc1, qc2, qc3]).result()
        self.assertIsInstance(result, SamplerResult)
        self.assertEqual(len(result.quasi_dists), 4)

        keys, values = zip(*sorted(result.quasi_dists[0].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [1, 0, 0, 0])

        keys, values = zip(*sorted(result.quasi_dists[1].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 1, 0, 0])

        keys, values = zip(*sorted(result.quasi_dists[2].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 0, 1, 0])

        keys, values = zip(*sorted(result.quasi_dists[3].items()))
        self.assertTupleEqual(keys, tuple(range(4)))
        np.testing.assert_allclose(values, [0, 0, 0, 1])

    def test_run_single_circuit(self):
        """Test for single circuit case."""

        sampler = Sampler()

        with self.subTest("No parameter"):
            circuit = self._circuit[1]
            target = self._target[1]
            param_vals = [None, [], [[]], np.array([]), np.array([[]])]
            for val in param_vals:
                with self.subTest(f"{circuit.name} w/ {val}"):
                    result = sampler.run(circuit, val).result()
                    self._compare_probs(result.quasi_dists, target)
                    self.assertEqual(len(result.metadata), 1)

        with self.subTest("One parameter"):
            circuit = QuantumCircuit(1, 1, name="X gate")
            param = Parameter("x")
            circuit.ry(param, 0)
            circuit.measure(0, 0)
            target = [{1: 1}]
            param_vals = [
                [np.pi],
                [[np.pi]],
                np.array([np.pi]),
                np.array([[np.pi]]),
                [np.array([np.pi])],
            ]
            for val in param_vals:
                with self.subTest(f"{circuit.name} w/ {val}"):
                    result = sampler.run(circuit, val).result()
                    self._compare_probs(result.quasi_dists, target)
                    self.assertEqual(len(result.metadata), 1)

        with self.subTest("More than one parameter"):
            circuit = self._pqc
            target = [self._pqc_target[0]]
            param_vals = [
                self._pqc_params[0],
                [self._pqc_params[0]],
                np.array(self._pqc_params[0]),
                np.array([self._pqc_params[0]]),
                [np.array(self._pqc_params[0])],
            ]
            for val in param_vals:
                with self.subTest(f"{circuit.name} w/ {val}"):
                    result = sampler.run(circuit, val).result()
                    self._compare_probs(result.quasi_dists, target)
                    self.assertEqual(len(result.metadata), 1)

    def test_run_errors(self):
        """Test for errors with run method"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()
        qc3 = QuantumCircuit(1)
        qc4 = QuantumCircuit(1, 1)

        sampler = Sampler()
        with self.subTest("set parameter values to a non-parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc1], [[1e2]])
        with self.subTest("missing all parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc2], [[]])
        with self.subTest("missing some parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc2], [[1e2]])
        with self.subTest("too many parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc2], [[1e2]] * 100)
        with self.subTest("no classical bits"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc3], [[]])
        with self.subTest("no measurement"):
            with self.assertRaises(QiskitError):
                # The following raises QiskitError because this check is located in
                # `Sampler._preprocess_circuit`
                _ = sampler.run([qc4], [[]])

    def test_run_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        sampler = Sampler()
        with self.subTest("one circuit"):
            result = sampler.run([qc], shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 1)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 1)

        with self.subTest("two circuits"):
            result = sampler.run([qc, qc], shots=1000).result()
            self.assertEqual(len(result.quasi_dists), 2)
            for q_d in result.quasi_dists:
                quasi_dist = {k: v for k, v in q_d.items() if v != 0.0}
                self.assertDictEqual(quasi_dist, {0: 1.0})
            self.assertEqual(len(result.metadata), 2)

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        qc.measure_all()
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        sampler = Sampler()
        target = sampler.run([qc] * k, params_list).result()

        with self.subTest("ndarrary"):
            result = sampler.run([qc] * k, params_array).result()
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictEqual(result.quasi_dists[i], target.quasi_dists[i])

        with self.subTest("list of ndarray"):
            result = sampler.run([qc] * k, params_list_array).result()
            self.assertEqual(len(result.metadata), k)
            for i in range(k):
                self.assertDictEqual(result.quasi_dists[i], target.quasi_dists[i])

    def test_run_with_shots_option(self):
        """test with shots option."""
        params, target = self._generate_params_target([1])
        sampler = Sampler()
        result = sampler.run(
            circuits=[self._pqc], parameter_values=params, shots=1024, seed=15
        ).result()
        self._compare_probs(result.quasi_dists, target)

    def test_run_with_shots_option_none(self):
        """test with shots=None option. Seed is ignored then."""
        sampler = Sampler()
        result_42 = sampler.run(
            [self._pqc], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=42
        ).result()
        result_15 = sampler.run(
            [self._pqc], parameter_values=[[0, 1, 1, 2, 3, 5]], shots=None, seed=15
        ).result()
        self.assertDictAlmostEqual(result_42.quasi_dists, result_15.quasi_dists)

    def test_primitive_job_status_done(self):
        """test primitive job's status"""
        bell = self._circuit[1]
        sampler = Sampler()
        job = sampler.run(circuits=[bell])
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_options(self):
        """Test for options"""
        with self.subTest("init"):
            sampler = Sampler(options={"shots": 3000})
            self.assertEqual(sampler.options.get("shots"), 3000)
        with self.subTest("set_options"):
            sampler.set_options(shots=1024, seed=15)
            self.assertEqual(sampler.options.get("shots"), 1024)
            self.assertEqual(sampler.options.get("seed"), 15)
        with self.subTest("run"):
            params, target = self._generate_params_target([1])
            result = sampler.run([self._pqc], parameter_values=params).result()
            self._compare_probs(result.quasi_dists, target)
            self.assertEqual(result.quasi_dists[0].shots, 1024)

    def test_different_circuits(self):
        """Test collision of quantum circuits."""

        with self.subTest("Ry circuit"):

            def test_func(n):
                qc = QuantumCircuit(1, 1, name="foo")
                qc.ry(n, 0)
                return qc

            keys = [_circuit_key(test_func(i)) for i in range(5)]
            self.assertEqual(len(keys), len(set(keys)))

        with self.subTest("pulse circuit"):

            def test_with_scheduling(n):
                custom_gate = pulse.Schedule(name="custom_x_gate")
                custom_gate.insert(
                    0, pulse.Play(pulse.Constant(160 * n, 0.1), pulse.DriveChannel(0)), inplace=True
                )
                qc = QuantumCircuit(1)
                qc.x(0)
                qc.add_calibration("x", qubits=(0,), schedule=custom_gate)
                return transpile(qc, FakeAlmaden(), scheduling_method="alap")

            keys = [_circuit_key(test_with_scheduling(i)) for i in range(1, 5)]
            self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
