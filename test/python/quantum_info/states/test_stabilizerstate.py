# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for Stabilizerstate quantum state class."""

import unittest
from test import combine
import logging
from ddt import ddt

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit

from qiskit.quantum_info.random import random_clifford, random_pauli
from qiskit.quantum_info.states import StabilizerState, Statevector
from qiskit.circuit.library import IGate, XGate, HGate
from qiskit.quantum_info.operators import Clifford, Pauli, Operator


logger = logging.getLogger(__name__)


@ddt
class TestStabilizerState(QiskitTestCase):
    """Tests for StabilizerState class."""

    rng = np.random.default_rng(12345)
    samples = 10
    shots = 1000
    threshold = 0.1 * shots

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_clifford(self, num_qubits):
        """Test initialization from Clifford."""
        stab1 = StabilizerState(random_clifford(num_qubits, seed=self.rng))
        stab2 = StabilizerState(stab1)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_circuit(self, num_qubits):
        """Test initialization from a Clifford circuit."""
        cliff = random_clifford(num_qubits, seed=self.rng)
        stab1 = StabilizerState(cliff.to_circuit())
        stab2 = StabilizerState(cliff)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_instruction(self, num_qubits):
        """Test initialization from a Clifford instruction."""
        cliff = random_clifford(num_qubits, seed=self.rng)
        stab1 = StabilizerState(cliff.to_instruction())
        stab2 = StabilizerState(cliff)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_pauli(self, num_qubits):
        """Test initialization from pauli."""
        pauli = random_pauli(num_qubits, seed=self.rng)
        stab1 = StabilizerState(pauli)
        stab2 = StabilizerState(stab1)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_to_operator(self, num_qubits):
        """Test to_operator method for returning projector."""
        for _ in range(self.samples):
            stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
            target = Operator(stab)
            op = StabilizerState(stab).to_operator()
            self.assertEqual(op, target)

    @combine(num_qubits=[2, 3, 4])
    def test_trace(self, num_qubits):
        """Test trace methods"""
        stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
        trace = stab.trace()
        self.assertEqual(trace, 1.0)

    @combine(num_qubits=[2, 3, 4])
    def test_purity(self, num_qubits):
        """Test purity methods"""
        stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
        purity = stab.purity()
        self.assertEqual(purity, 1.0)

    @combine(num_qubits=[2, 3])
    def test_conjugate(self, num_qubits):
        """Test conjugate method."""
        for _ in range(self.samples):
            cliff = random_clifford(num_qubits, seed=self.rng)
            target = StabilizerState(cliff.conjugate())
            state = StabilizerState(cliff).conjugate()
            self.assertEqual(state, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(2, seed=self.rng)
            cliff2 = random_clifford(3, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.tensor(cliff2))
            state = stab1.tensor(stab2)
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(2, seed=self.rng)
            cliff2 = random_clifford(3, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.expand(cliff2))
            state = stab1.expand(stab2)
            self.assertEqual(state, target)

    @combine(num_qubits=[2, 3, 4])
    def test_evolve(self, num_qubits):
        """Test evolve method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(num_qubits, seed=self.rng)
            cliff2 = random_clifford(num_qubits, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.compose(cliff2))
            state = stab1.evolve(stab2)
            self.assertEqual(state, target)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_evolve_subsystem(self, num_qubits_1, num_qubits_2):
        """Test subsystem evolve method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(num_qubits_1, seed=self.rng)
            cliff2 = random_clifford(num_qubits_2, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            target = StabilizerState(cliff1.compose(cliff2, qargs))
            state = stab1.evolve(stab2, qargs)
            self.assertEqual(state, target)

    def test_measure_single_qubit(self):
        """Test a measurement of a single qubit"""
        for _ in range(self.samples):
            cliff = Clifford(XGate())
            stab = StabilizerState(cliff)
            value = stab.measure()[0]
            self.assertEqual(value, "1")

            cliff = Clifford(IGate())
            stab = StabilizerState(cliff)
            value = stab.measure()[0]
            self.assertEqual(value, "0")

            cliff = Clifford(HGate())
            stab = StabilizerState(cliff)
            value = stab.measure()[0]
            self.assertIn(value, ["0", "1"])

    def test_measure_qubits(self):
        """Test a measurement of a subsystem of qubits"""

        for _ in range(self.samples):
            num_qubits = 4
            qc = QuantumCircuit(num_qubits)
            stab = StabilizerState(qc)
            value = stab.measure()[0]
            self.assertEqual(value, "0000")
            value = stab.measure([0, 2])[0]
            self.assertEqual(value, "00")
            value = stab.measure([1])[0]
            self.assertEqual(value, "0")

            for i in range(num_qubits):
                qc.x(i)
            stab = StabilizerState(qc)
            value = stab.measure()[0]
            self.assertEqual(value, "1111")
            value = stab.measure([2, 0])[0]
            self.assertEqual(value, "11")
            value = stab.measure([1])[0]
            self.assertEqual(value, "1")

            qc = QuantumCircuit(num_qubits)
            qc.h(0)
            stab = StabilizerState(qc)
            value = stab.measure()[0]
            self.assertIn(value, ["0000", "0001"])
            value = stab.measure([0, 1])[0]
            self.assertIn(value, ["00", "01"])
            value = stab.measure([2])[0]
            self.assertEqual(value, "0")

            qc = QuantumCircuit(num_qubits)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            qc.cx(0, 3)
            stab = StabilizerState(qc)
            value = stab.measure()[0]
            self.assertIn(value, ["0000", "1111"])
            value = stab.measure([3, 1])[0]
            self.assertIn(value, ["00", "11"])
            value = stab.measure([2])[0]
            self.assertIn(value, ["0", "1"])

    def test_reset_single_qubit(self):
        """Test reset method of a single qubit"""

        empty_qc = QuantumCircuit(1)

        for _ in range(self.samples):
            cliff = Clifford(XGate())
            stab = StabilizerState(cliff)
            value = stab.reset([0])
            target = StabilizerState(empty_qc)
            self.assertEqual(value, target)

            cliff = Clifford(HGate())
            stab = StabilizerState(cliff)
            value = stab.reset([0])
            target = StabilizerState(empty_qc)
            self.assertEqual(value, target)

    def test_reset_qubits(self):
        """Test reset method of a subsystem of qubits"""

        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)

        for _ in range(self.samples):
            with self.subTest(msg="reset (None)"):
                stab = StabilizerState(qc)
                res = stab.reset()
                value = res.measure()[0]
                self.assertEqual(value, "000")

        for _ in range(self.samples):
            for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
                with self.subTest(msg=f"reset (qargs={qargs})"):
                    stab = StabilizerState(qc)
                    res = stab.reset(qargs)
                    value = res.measure()[0]
                    self.assertEqual(value, "000")

        for _ in range(self.samples):
            with self.subTest(msg="reset ([0])"):
                stab = StabilizerState(qc)
                res = stab.reset([0])
                value = res.measure()[0]
                self.assertIn(value, ["000", "110"])

        for _ in range(self.samples):
            with self.subTest(msg="reset ([1])"):
                stab = StabilizerState(qc)
                res = stab.reset([1])
                value = res.measure()[0]
                self.assertIn(value, ["000", "101"])

        for _ in range(self.samples):
            with self.subTest(msg="reset ([2])"):
                stab = StabilizerState(qc)
                res = stab.reset([2])
                value = res.measure()[0]
                self.assertIn(value, ["000", "011"])

        for _ in range(self.samples):
            for qargs in [[0, 1], [1, 0]]:
                with self.subTest(msg=f"reset (qargs={qargs})"):
                    stab = StabilizerState(qc)
                    res = stab.reset(qargs)
                    value = res.measure()[0]
                    self.assertIn(value, ["000", "100"])

        for _ in range(self.samples):
            for qargs in [[0, 2], [2, 0]]:
                with self.subTest(msg=f"reset (qargs={qargs})"):
                    stab = StabilizerState(qc)
                    res = stab.reset(qargs)
                    value = res.measure()[0]
                    self.assertIn(value, ["000", "010"])

        for _ in range(self.samples):
            for qargs in [[1, 2], [2, 1]]:
                with self.subTest(msg=f"reset (qargs={qargs})"):
                    stab = StabilizerState(qc)
                    res = stab.reset(qargs)
                    value = res.measure()[0]
                    self.assertIn(value, ["000", "001"])

    def test_probablities_dict_single_qubit(self):
        """Test probabilities and probabilities_dict methods of a single qubit"""

        num_qubits = 1
        qc = QuantumCircuit(num_qubits)

        for _ in range(self.samples):
            with self.subTest(msg="P(id(0))"):
                stab = StabilizerState(qc)
                value = stab.probabilities_dict()
                target = {"0": 1}
                self.assertEqual(value, target)
                probs = stab.probabilities()
                target = np.array([1, 0])
                self.assertTrue(np.allclose(probs, target))

        qc.x(0)
        for _ in range(self.samples):
            with self.subTest(msg="P(x(0))"):
                stab = StabilizerState(qc)
                value = stab.probabilities_dict()
                target = {"1": 1}
                self.assertEqual(value, target)
                probs = stab.probabilities()
                target = np.array([0, 1])
                self.assertTrue(np.allclose(probs, target))

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for _ in range(self.samples):
            with self.subTest(msg="P(h(0))"):
                stab = StabilizerState(qc)
                value = stab.probabilities_dict()
                target = {"0": 0.5, "1": 0.5}
                self.assertEqual(value, target)
                probs = stab.probabilities()
                target = np.array([0.5, 0.5])
                self.assertTrue(np.allclose(probs, target))

    def test_probablities_dict_two_qubits(self):
        """Test probabilities and probabilities_dict methods of two qubits"""

        num_qubits = 2
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        stab = StabilizerState(qc)

        for _ in range(self.samples):
            with self.subTest(msg="P(None)"):
                value = stab.probabilities_dict()
                target = {"00": 0.5, "01": 0.5}
                self.assertEqual(value, target)
                probs = stab.probabilities()
                target = np.array([0.5, 0.5, 0, 0])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P([0, 1])"):
                value = stab.probabilities_dict([0, 1])
                target = {"00": 0.5, "01": 0.5}
                self.assertEqual(value, target)
                probs = stab.probabilities([0, 1])
                target = np.array([0.5, 0.5, 0, 0])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P([1, 0])"):
                value = stab.probabilities_dict([1, 0])
                target = {"00": 0.5, "10": 0.5}
                self.assertEqual(value, target)
                probs = stab.probabilities([1, 0])
                target = np.array([0.5, 0, 0.5, 0])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P[0]"):
                value = stab.probabilities_dict([0])
                target = {"0": 0.5, "1": 0.5}
                self.assertEqual(value, target)
                probs = stab.probabilities([0])
                target = np.array([0.5, 0.5])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P([1])"):
                value = stab.probabilities_dict([1])
                target = {"0": 1.0}
                self.assertEqual(value, target)
                probs = stab.probabilities([1])
                target = np.array([1, 0])
                self.assertTrue(np.allclose(probs, target))

    def test_probablities_dict_qubits(self):
        """Test probabilities and probabilities_dict methods of a subsystem of qubits"""

        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        stab = StabilizerState(qc)

        for _ in range(self.samples):
            with self.subTest(msg="P(None), decimals=1"):
                value = stab.probabilities_dict(decimals=1)
                target = {
                    "000": 0.1,
                    "001": 0.1,
                    "010": 0.1,
                    "011": 0.1,
                    "100": 0.1,
                    "101": 0.1,
                    "110": 0.1,
                    "111": 0.1,
                }
                self.assertEqual(value, target)
                probs = stab.probabilities(decimals=1)
                target = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P(None), decimals=2"):
                value = stab.probabilities_dict(decimals=2)
                target = {
                    "000": 0.12,
                    "001": 0.12,
                    "010": 0.12,
                    "011": 0.12,
                    "100": 0.12,
                    "101": 0.12,
                    "110": 0.12,
                    "111": 0.12,
                }
                self.assertEqual(value, target)
                probs = stab.probabilities(decimals=2)
                target = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
                self.assertTrue(np.allclose(probs, target))

        for _ in range(self.samples):
            with self.subTest(msg="P(None), decimals=3"):
                value = stab.probabilities_dict(decimals=3)
                target = {
                    "000": 0.125,
                    "001": 0.125,
                    "010": 0.125,
                    "011": 0.125,
                    "100": 0.125,
                    "101": 0.125,
                    "110": 0.125,
                    "111": 0.125,
                }
                self.assertEqual(value, target)
                probs = stab.probabilities(decimals=3)
                target = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
                self.assertTrue(np.allclose(probs, target))

    def test_probablities_dict_ghz(self):
        """Test probabilities and probabilities_dict method of a subsystem of qubits"""

        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)

        with self.subTest(msg="P(None)"):
            stab = StabilizerState(qc)
            value = stab.probabilities_dict()
            target = {"000": 0.5, "111": 0.5}
            self.assertEqual(value, target)
            probs = stab.probabilities()
            target = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
            self.assertTrue(np.allclose(probs, target))

        # 3-qubit qargs
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = stab.probabilities_dict(qargs)
                target = {"000": 0.5, "111": 0.5}
                self.assertDictAlmostEqual(probs, target)
                probs = stab.probabilities(qargs)
                target = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
                self.assertTrue(np.allclose(probs, target))

        # 2-qubit qargs
        for qargs in [[0, 1], [2, 1], [1, 0], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = stab.probabilities_dict(qargs)
                target = {"00": 0.5, "11": 0.5}
                self.assertDictAlmostEqual(probs, target)
                probs = stab.probabilities(qargs)
                target = np.array([0.5, 0, 0, 0.5])
                self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = stab.probabilities_dict(qargs)
                target = {"0": 0.5, "1": 0.5}
                self.assertDictAlmostEqual(probs, target)
                probs = stab.probabilities(qargs)
                target = np.array([0.5, 0.5])
                self.assertTrue(np.allclose(probs, target))

    @combine(num_qubits=[2, 3, 4])
    def test_probs_random_subsystem(self, num_qubits):
        """Test probabilities and probabilities_dict methods
        of random cliffords for a subsystem of qubits"""

        for _ in range(self.samples):
            for subsystem_size in range(1, num_qubits):
                cliff = random_clifford(num_qubits, seed=self.rng)
                qargs = np.random.choice(num_qubits, size=subsystem_size, replace=False)
                qc = cliff.to_circuit()
                stab = StabilizerState(cliff)
                probs = stab.probabilities(qargs)
                probs_dict = stab.probabilities_dict(qargs)
                target = Statevector(qc).probabilities(qargs)
                target_dict = Statevector(qc).probabilities_dict(qargs)
                self.assertTrue(np.allclose(probs, target))
                self.assertDictAlmostEqual(probs_dict, target_dict)

    def test_expval_single_qubit(self):
        """Test expectation_value method of a single qubit"""

        num_qubits = 1

        qc = QuantumCircuit(num_qubits)
        stab = StabilizerState(qc)
        pairs = [("Z", 1), ("X", 0), ("Y", 0), ("I", 1)]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.x(0)
        stab = StabilizerState(qc)
        pairs = [("Z", -1), ("X", 0), ("Y", 0), ("I", 1)]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        stab = StabilizerState(qc)
        pairs = [("Z", 0), ("X", 1), ("Y", 0), ("I", 1)]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

    def test_expval_two_qubits(self):
        """Test expectation_value method of two qubits"""

        num_qubits = 2

        qc = QuantumCircuit(num_qubits)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 0),
            ("YY", 0),
            ("ZZ", 1),
            ("IX", 0),
            ("IY", 0),
            ("IZ", 1),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.x(0)
        qc.x(1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 0),
            ("YY", 0),
            ("ZZ", 1),
            ("IX", 0),
            ("IY", 0),
            ("IZ", -1),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.h(1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 1),
            ("YY", 0),
            ("ZZ", 0),
            ("IX", 1),
            ("IY", 0),
            ("IZ", 0),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.x(0)
        qc.h(1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 0),
            ("YY", 0),
            ("ZZ", 0),
            ("IX", 0),
            ("IY", 0),
            ("IZ", -1),
            ("XY", 0),
            ("XZ", -1),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 1),
            ("YY", -1),
            ("ZZ", 1),
            ("IX", 0),
            ("IY", 0),
            ("IZ", 0),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 1),
            ("YY", 1),
            ("ZZ", -1),
            ("IX", 0),
            ("IY", 0),
            ("IZ", 0),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.sdg(0)
        qc.sdg(1)
        qc.h(0)
        qc.h(1)
        stab = StabilizerState(qc)
        pairs = [
            ("II", 1),
            ("XX", 1),
            ("YY", 1),
            ("ZZ", -1),
            ("IX", 0),
            ("IY", 0),
            ("IZ", 0),
            ("XY", 0),
            ("XZ", 0),
            ("YZ", 0),
        ]
        for label, target in pairs:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = stab.expectation_value(op)
                self.assertEqual(expval, target)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_expval_random(self, num_qubits):
        """Test expectation_value method of random Cliffords"""

        for _ in range(self.samples):
            cliff = random_clifford(num_qubits, seed=self.rng)
            op = random_pauli(num_qubits, seed=self.rng)
            qc = cliff.to_circuit()
            stab = StabilizerState(cliff)
            exp_val = stab.expectation_value(op)
            target = Statevector(qc).expectation_value(op)
            self.assertAlmostEqual(exp_val, target)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_expval_random_subsystem(self, num_qubits):
        """Test expectation_value method of random Cliffords and a subsystem"""

        for _ in range(self.samples):
            cliff = random_clifford(num_qubits, seed=self.rng)
            op = random_pauli(2, seed=self.rng)
            qargs = np.random.choice(num_qubits, size=2, replace=False)
            qc = cliff.to_circuit()
            stab = StabilizerState(cliff)
            exp_val = stab.expectation_value(op, qargs)
            target = Statevector(qc).expectation_value(op, qargs)
            self.assertAlmostEqual(exp_val, target)

    def test_sample_counts_reset_bell(self):
        """Test sample_counts after reset for Bell state"""

        num_qubits = 2
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        stab = StabilizerState(qc)

        target = {"00": self.shots / 2, "10": self.shots / 2}
        counts = {"00": 0, "10": 0}
        for _ in range(self.shots):
            res = stab.reset([0])
            value = res.measure()[0]
            counts[value] += 1
        self.assertDictAlmostEqual(counts, target, self.threshold)

        target = {"00": self.shots / 2, "01": self.shots / 2}
        counts = {"00": 0, "01": 0}
        for _ in range(self.shots):
            res = stab.reset([1])
            value = res.measure()[0]
            counts[value] += 1
        self.assertDictAlmostEqual(counts, target, self.threshold)

    def test_sample_counts_memory_ghz(self):
        """Test sample_counts and sample_memory method for GHZ state"""

        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        stab = StabilizerState(qc)

        # 3-qubit qargs
        target = {"000": self.shots / 2, "111": self.shots / 2}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {"00": self.shots / 2, "11": self.shots / 2}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 0]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": self.shots / 2, "1": self.shots / 2}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))

    def test_sample_counts_memory_superposition(self):
        """Test sample_counts and sample_memory method of a 3-qubit superposition"""

        num_qubits = 3
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        stab = StabilizerState(qc)

        # 3-qubit qargs
        target = {
            "000": self.shots / 8,
            "001": self.shots / 8,
            "010": self.shots / 8,
            "011": self.shots / 8,
            "100": self.shots / 8,
            "101": self.shots / 8,
            "110": self.shots / 8,
            "111": self.shots / 8,
        }
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {
            "00": self.shots / 4,
            "01": self.shots / 4,
            "10": self.shots / 4,
            "11": self.shots / 4,
        }
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 0]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": self.shots / 2, "1": self.shots / 2}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = stab.sample_counts(self.shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, self.threshold)

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = stab.sample_memory(self.shots, qargs=qargs)
                self.assertEqual(len(memory), self.shots)
                self.assertEqual(set(memory), set(target))


if __name__ == "__main__":
    unittest.main()
