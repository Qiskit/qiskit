# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Sampler V2."""

from __future__ import annotations

import unittest
from dataclasses import astuple

import numpy as np
from numpy.typing import NDArray

from qiskit import ClassicalRegister, QiskitError, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, UnitaryGate
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit.primitives.containers import BitArray
from qiskit.primitives.containers.data_bin import DataBin
from qiskit.primitives.statevector_sampler import Sampler
from qiskit.providers import JobStatus, JobV1
from qiskit.test import QiskitTestCase
from qiskit.utils.optionals import HAS_PYDANTIC


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed.")
class TestSampler(QiskitTestCase):
    """Test Sampler"""

    def setUp(self):
        super().setUp()
        self._shots = 10000
        self._options = {"execution": {"shots": self._shots, "seed": 123}}

        self._cases = []
        hadamard = QuantumCircuit(1, 1, name="Hadamard")
        hadamard.h(0)
        hadamard.measure(0, 0)
        self._cases.append((hadamard, None, {0: 5000, 1: 5000}))  # case 0

        bell = QuantumCircuit(2, name="Bell")
        bell.h(0)
        bell.cx(0, 1)
        bell.measure_all()
        self._cases.append((bell, None, {0: 5000, 3: 5000}))  # case 1

        pqc = RealAmplitudes(num_qubits=2, reps=2)
        pqc.measure_all()
        self._cases.append((pqc, [0] * 6, {0: 10000}))  # case 2
        self._cases.append((pqc, [1] * 6, {0: 168, 1: 3389, 2: 470, 3: 5973}))  # case 3
        self._cases.append((pqc, [0, 1, 1, 2, 3, 5], {0: 1339, 1: 3534, 2: 912, 3: 4215}))  # case 4
        self._cases.append((pqc, [1, 2, 3, 4, 5, 6], {0: 634, 1: 291, 2: 6039, 3: 3036}))  # case 5

        pqc2 = RealAmplitudes(num_qubits=2, reps=3)
        pqc2.measure_all()
        self._cases.append(
            (pqc2, [0, 1, 2, 3, 4, 5, 6, 7], {0: 1898, 1: 6864, 2: 928, 3: 311})
        )  # case 6

    def _assert_allclose(self, bitarray: BitArray, target: NDArray | BitArray, rtol=1e-1):
        self.assertEqual(bitarray.shape, target.shape)
        for idx in np.ndindex(bitarray.shape):
            int_counts = bitarray.get_int_counts(idx)
            target_counts = (
                target.get_int_counts(idx) if isinstance(target, BitArray) else target[idx]
            )
            max_key = max(max(int_counts.keys()), max(target_counts.keys()))
            ary = np.array([int_counts.get(i, 0) for i in range(max_key + 1)])
            tgt = np.array([target_counts.get(i, 0) for i in range(max_key + 1)])
            np.testing.assert_allclose(ary, tgt, rtol=rtol, err_msg=f"index: {idx}")

    def test_sampler_run(self):
        """Test Sampler.run()."""
        bell, _, target = self._cases[1]

        with self.subTest("single"):
            sampler = Sampler(options=self._options)
            job = sampler.run([bell])
            self.assertIsInstance(job, JobV1)
            result = job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertIsInstance(result.metadata, dict)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], PubResult)
            self.assertIsInstance(result[0].data, DataBin)
            self.assertIsInstance(result[0].data.meas, BitArray)
            self._assert_allclose(result[0].data.meas, np.array(target))

        with self.subTest("single with param"):
            sampler = Sampler(options=self._options)
            job = sampler.run([(bell, ())])
            self.assertIsInstance(job, JobV1)
            result = job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertIsInstance(result.metadata, dict)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], PubResult)
            self.assertIsInstance(result[0].data, DataBin)
            self.assertIsInstance(result[0].data.meas, BitArray)
            self._assert_allclose(result[0].data.meas, np.array(target))

        with self.subTest("single array"):
            sampler = Sampler(options=self._options)
            job = sampler.run([(bell, [()])])
            self.assertIsInstance(job, JobV1)
            result = job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertIsInstance(result.metadata, dict)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], PubResult)
            self.assertIsInstance(result[0].data, DataBin)
            self.assertIsInstance(result[0].data.meas, BitArray)
            self._assert_allclose(result[0].data.meas, np.array([target]))

        with self.subTest("multiple"):
            sampler = Sampler(options=self._options)
            job = sampler.run([(bell, [(), (), ()])])
            self.assertIsInstance(job, JobV1)
            result = job.result()
            self.assertIsInstance(result, PrimitiveResult)
            self.assertIsInstance(result.metadata, dict)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], PubResult)
            self.assertIsInstance(result[0].data, DataBin)
            self.assertIsInstance(result[0].data.meas, BitArray)
            self._assert_allclose(result[0].data.meas, np.array([target, target, target]))

    def test_sample_run_multiple_circuits(self):
        """Test Sampler.run() with multiple circuits."""
        bell, _, target = self._cases[1]
        sampler = Sampler(options=self._options)
        result = sampler.run([bell, bell, bell]).result()
        self.assertEqual(len(result), 3)
        self._assert_allclose(result[0].data.meas, np.array(target))
        self._assert_allclose(result[1].data.meas, np.array(target))
        self._assert_allclose(result[2].data.meas, np.array(target))

    def test_sampler_run_with_parameterized_circuits(self):
        """Test Sampler.run() with parameterized circuits."""

        pqc1, param1, target1 = self._cases[4]
        pqc2, param2, target2 = self._cases[5]
        pqc3, param3, target3 = self._cases[6]

        sampler = Sampler(options=self._options)
        result = sampler.run([(pqc1, param1), (pqc2, param2), (pqc3, param3)]).result()
        self.assertEqual(len(result), 3)
        self._assert_allclose(result[0].data.meas, np.array(target1))
        self._assert_allclose(result[1].data.meas, np.array(target2))
        self._assert_allclose(result[2].data.meas, np.array(target3))

    def test_run_1qubit(self):
        """test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc.measure_all()
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        qc2.measure_all()

        sampler = Sampler(options=self._options)
        result = sampler.run([qc, qc2]).result()
        self.assertEqual(len(result), 2)
        for i in range(2):
            self._assert_allclose(result[i].data.meas, np.array({i: self._shots}))

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

        sampler = Sampler(options=self._options)
        result = sampler.run([qc0, qc1, qc2, qc3]).result()
        self.assertEqual(len(result), 4)
        for i in range(4):
            self._assert_allclose(result[i].data.meas, np.array({i: self._shots}))

    def test_run_single_circuit(self):
        """Test for single circuit case."""

        sampler = Sampler(options=self._options)

        with self.subTest("No parameter"):
            circuit, _, target = self._cases[1]
            param_target = [
                (None, np.array(target)),
                ((), np.array(target)),
                ([], np.array(target)),
                (np.array([]), np.array(target)),
                (((),), np.array([target])),
                (([],), np.array([target])),
                ([[]], np.array([target])),
                ([()], np.array([target])),
                (np.array([[]]), np.array([target])),
            ]
            for param, target in param_target:
                with self.subTest(f"{circuit.name} w/ {param}"):
                    result = sampler.run([(circuit, param)]).result()
                    self.assertEqual(len(result), 1)
                    self._assert_allclose(result[0].data.meas, target)

        with self.subTest("One parameter"):
            circuit = QuantumCircuit(1, 1, name="X gate")
            param = Parameter("x")
            circuit.ry(param, 0)
            circuit.measure(0, 0)
            param_target = [
                ([np.pi], np.array({1: self._shots})),
                ((np.pi,), np.array({1: self._shots})),
                (np.array([np.pi]), np.array({1: self._shots})),
                ([[np.pi]], np.array([{1: self._shots}])),
                (((np.pi,),), np.array([{1: self._shots}])),
                (np.array([[np.pi]]), np.array([{1: self._shots}])),
            ]
            for param, target in param_target:
                with self.subTest(f"{circuit.name} w/ {param}"):
                    result = sampler.run([(circuit, param)]).result()
                    self.assertEqual(len(result), 1)
                    self._assert_allclose(result[0].data.c, target)

        with self.subTest("More than one parameter"):
            circuit, param, target = self._cases[3]
            param_target = [
                (param, np.array(target)),
                (tuple(param), np.array(target)),
                (np.array(param), np.array(target)),
                ((param,), np.array([target])),
                ([param], np.array([target])),
                (np.array([param]), np.array([target])),
            ]
            for param, target in param_target:
                with self.subTest(f"{circuit.name} w/ {param}"):
                    result = sampler.run([(circuit, param)]).result()
                    self.assertEqual(len(result), 1)
                    self._assert_allclose(result[0].data.meas, target)

    def test_run_reverse_meas_order(self):
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

        sampler = Sampler(options=self._options)
        result = sampler.run([(qc, [0, 0]), (qc, [np.pi / 2, 0])]).result()
        self.assertEqual(len(result), 2)

        # qc({x: 0, y: 0})
        self._assert_allclose(result[0].data.c, np.array({1: self._shots}))

        # qc({x: pi/2, y: 0})
        self._assert_allclose(result[1].data.c, np.array({1: self._shots / 2, 5: self._shots / 2}))

    def test_run_errors(self):
        """Test for errors with run method"""
        qc1 = QuantumCircuit(1)
        qc1.measure_all()
        qc2 = RealAmplitudes(num_qubits=1, reps=1)
        qc2.measure_all()
        qc3 = QuantumCircuit(1)
        qc4 = QuantumCircuit(1, 1)
        with qc4.for_loop(range(5)):
            qc4.h(0)

        sampler = Sampler(options=self._options)
        with self.subTest("set parameter values to a non-parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([(qc1, [1e2])]).result()
        with self.subTest("missing all parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc2]).result()
            with self.assertRaises(ValueError):
                _ = sampler.run([(qc2, [])]).result()
            with self.assertRaises(ValueError):
                _ = sampler.run([(qc2, None)]).result()
        with self.subTest("missing some parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([(qc2, [1e2])]).result()
        with self.subTest("too many parameter values for a parameterized circuit"):
            with self.assertRaises(ValueError):
                _ = sampler.run([(qc2, [1e2] * 100)]).result()
        with self.subTest("no classical bits"):
            with self.assertRaises(ValueError):
                _ = sampler.run([qc3]).result()
        with self.subTest("with control flow"):
            with self.assertRaises(QiskitError):
                _ = sampler.run([qc4]).result()

    def test_run_empty_parameter(self):
        """Test for empty parameter"""
        n = 5
        qc = QuantumCircuit(n, n - 1)
        qc.measure(range(n - 1), range(n - 1))
        sampler = Sampler(options=self._options)
        with self.subTest("one circuit"):
            result = sampler.run([qc]).result()
            self.assertEqual(len(result), 1)
            self._assert_allclose(result[0].data.c, np.array({0: self._shots}))

        with self.subTest("two circuits"):
            result = sampler.run([qc, qc]).result()
            self.assertEqual(len(result), 2)
            for i in range(2):
                self._assert_allclose(result[i].data.c, np.array({0: self._shots}))

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = RealAmplitudes(num_qubits=2, reps=2)
        qc.measure_all()
        k = 5
        params_array = np.random.rand(k, qc.num_parameters)
        params_list = params_array.tolist()
        sampler = Sampler(options=self._options)
        target = sampler.run([(qc, params_list)]).result()

        with self.subTest("ndarray"):
            result = sampler.run([(qc, params_array)]).result()
            self.assertEqual(len(result), 1)
            self._assert_allclose(result[0].data.meas, target[0].data.meas)

        with self.subTest("split a list"):
            result = sampler.run([(qc, params) for params in params_list]).result()
            self.assertEqual(len(result), k)
            for i in range(k):
                self._assert_allclose(
                    result[i].data.meas, np.array(target[0].data.meas.get_int_counts(i))
                )

    def test_run_with_shots_option(self):
        """test with shots option."""
        bell, _, _ = self._cases[1]
        sampler = Sampler()
        shots = 100
        sampler.options.execution.shots = shots
        result = sampler.run([bell]).result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].data.meas.num_samples, shots)
        self.assertEqual(sum(result[0].data.meas.get_counts().values()), shots)

    def test_run_shots_result_size(self):
        """test with shots option to validate the result size"""
        n = 10
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.measure_all()
        sampler = Sampler(options=self._options)
        result = sampler.run([qc]).result()
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0].data.meas.num_samples, self._shots)
        self.assertEqual(sum(result[0].data.meas.get_counts().values()), self._shots)

    def test_primitive_job_status_done(self):
        """test primitive job's status"""
        bell, _, _ = self._cases[1]
        sampler = Sampler(options=self._options)
        job = sampler.run([bell])
        _ = job.result()
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_options(self):
        """Test for options"""
        with self.subTest("init"):
            sampler = Sampler(options={"execution": {"shots": 3000}})
            self.assertEqual(sampler.options.execution.shots, 3000)
        with self.subTest("set options"):
            sampler.options.execution.shots = 1024
            sampler.options.execution.seed = 15
            self.assertEqual(sampler.options.execution.shots, 1024)
            self.assertEqual(sampler.options.execution.seed, 15)
        with self.subTest("update options"):
            sampler.options.execution.update({"shots": 100, "seed": 12})
            self.assertEqual(sampler.options.execution.shots, 100)
            self.assertEqual(sampler.options.execution.seed, 12)

    def test_circuit_with_unitary(self):
        """Test for circuit with unitary gate."""

        with self.subTest("identity"):
            gate = UnitaryGate(np.eye(2))

            circuit = QuantumCircuit(1)
            circuit.append(gate, [0])
            circuit.measure_all()

            sampler = Sampler(options=self._options)
            result = sampler.run([circuit]).result()
            self.assertEqual(len(result), 1)
            self._assert_allclose(result[0].data.meas, np.array({0: self._shots}))

        with self.subTest("X"):
            gate = UnitaryGate([[0, 1], [1, 0]])

            circuit = QuantumCircuit(1)
            circuit.append(gate, [0])
            circuit.measure_all()

            sampler = Sampler(options=self._options)
            result = sampler.run([circuit]).result()
            self.assertEqual(len(result), 1)
            self._assert_allclose(result[0].data.meas, np.array({1: self._shots}))

    def test_metadata(self):
        """Test for metatdata."""
        qc, _, _ = self._cases[1]
        sampler = Sampler(options=self._options)
        result = sampler.run([qc]).result()
        self.assertEqual(len(result), 1)
        self.assertIn("shots", result[0].metadata)
        self.assertEqual(result[0].metadata["shots"], self._shots)

        shots = 100
        sampler.options.execution.shots = 100
        result = sampler.run([qc]).result()
        self.assertEqual(len(result), 1)
        self.assertIn("shots", result[0].metadata)
        self.assertEqual(result[0].metadata["shots"], shots)

    def test_circuit_with_multiple_cregs(self):
        """Test for circuit with multiple classical registers."""
        cases = []

        # case 1
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(2, "b")
        c = ClassicalRegister(3, "c")

        qc = QuantumCircuit(QuantumRegister(3), a, b, c)
        qc.h(range(3))
        qc.measure([0, 1, 2, 2], [0, 2, 4, 5])
        target = {"a": {0: 5000, 1: 5000}, "b": {0: 5000, 2: 5000}, "c": {0: 5000, 6: 5000}}
        cases.append(("use all cregs", qc, target))

        # case 2
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(5, "b")
        c = ClassicalRegister(3, "c")

        qc = QuantumCircuit(QuantumRegister(3), a, b, c)
        qc.h(range(3))
        qc.measure([0, 1, 2, 2], [0, 2, 4, 5])
        target = {
            "a": {0: 5000, 1: 5000},
            "b": {0: 2500, 2: 2500, 24: 2500, 26: 2500},
            "c": {0: 10000},
        }
        cases.append(("use only a and b", qc, target))

        # case 3
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(2, "b")
        c = ClassicalRegister(3, "c")

        qc = QuantumCircuit(QuantumRegister(3), a, b, c)
        qc.h(range(3))
        qc.measure(1, 5)
        target = {"a": {0: 10000}, "b": {0: 10000}, "c": {0: 5000, 4: 5000}}
        cases.append(("use only c", qc, target))

        # case 4
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(2, "b")
        c = ClassicalRegister(3, "c")

        qc = QuantumCircuit(QuantumRegister(3), a, b, c)
        qc.h(range(3))
        qc.measure([0, 1, 2], [5, 5, 5])
        target = {"a": {0: 10000}, "b": {0: 10000}, "c": {0: 5000, 4: 5000}}
        cases.append(("use only c multiple qubits", qc, target))

        # case 5
        a = ClassicalRegister(1, "a")
        b = ClassicalRegister(2, "b")
        c = ClassicalRegister(3, "c")

        qc = QuantumCircuit(QuantumRegister(3), a, b, c)
        qc.h(range(3))
        target = {"a": {0: 10000}, "b": {0: 10000}, "c": {0: 10000}}
        cases.append(("no measure", qc, target))

        for title, qc, target in cases:
            with self.subTest(title):
                sampler = Sampler(options=self._options)
                result = sampler.run([qc]).result()
                self.assertEqual(len(result), 1)
                data = result[0].data
                self.assertEqual(len(astuple(data)), 3)
                for creg in qc.cregs:
                    self.assertTrue(hasattr(data, creg.name))
                    self._assert_allclose(getattr(data, creg.name), np.array(target[creg.name]))


if __name__ == "__main__":
    unittest.main()
