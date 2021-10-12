# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
"""Tests for Clifford class."""

import unittest
from test import combine
from ddt import ddt

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit import Gate, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    CXGate,
    CZGate,
    SwapGate,
)
from qiskit.quantum_info.operators import Clifford, Operator
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_circuit
from qiskit.quantum_info.synthesis.clifford_decompose import (
    decompose_clifford_ag,
    decompose_clifford_bm,
    decompose_clifford_greedy,
)
from qiskit.quantum_info import random_clifford


class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(VGate(), [q[0]], []), (VGate(), [q[0]], [])]
        self.definition = qc


def random_clifford_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo random Clifford circuit."""

    if gates == "all":
        if num_qubits == 1:
            gates = ["i", "x", "y", "z", "h", "s", "sdg", "v", "w"]
        else:
            gates = ["i", "x", "y", "z", "h", "s", "sdg", "v", "w", "cx", "cz", "swap"]

    instructions = {
        "i": (IGate(), 1),
        "x": (XGate(), 1),
        "y": (YGate(), 1),
        "z": (ZGate(), 1),
        "h": (HGate(), 1),
        "s": (SGate(), 1),
        "sdg": (SdgGate(), 1),
        "v": (VGate(), 1),
        "w": (WGate(), 1),
        "cx": (CXGate(), 2),
        "cz": (CZGate(), 2),
        "swap": (SwapGate(), 2),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs)

    return circ


@ddt
class TestCliffordGates(QiskitTestCase):
    """Tests for clifford append gate functions."""

    def test_append_1_qubit_gate(self):
        """Tests for append of 1-qubit gates"""

        target_table = {
            "i": np.array([[[True, False], [False, True]]], dtype=bool),
            "id": np.array([[[True, False], [False, True]]], dtype=bool),
            "iden": np.array([[[True, False], [False, True]]], dtype=bool),
            "x": np.array([[[True, False], [False, True]]], dtype=bool),
            "y": np.array([[[True, False], [False, True]]], dtype=bool),
            "z": np.array([[[True, False], [False, True]]], dtype=bool),
            "h": np.array([[[False, True], [True, False]]], dtype=bool),
            "s": np.array([[[True, True], [False, True]]], dtype=bool),
            "sdg": np.array([[[True, True], [False, True]]], dtype=bool),
            "sinv": np.array([[[True, True], [False, True]]], dtype=bool),
            "v": np.array([[[True, True], [True, False]]], dtype=bool),
            "w": np.array([[[False, True], [True, True]]], dtype=bool),
        }

        target_phase = {
            "i": np.array([[False, False]], dtype=bool),
            "id": np.array([[False, False]], dtype=bool),
            "iden": np.array([[False, False]], dtype=bool),
            "x": np.array([[False, True]], dtype=bool),
            "y": np.array([[True, True]], dtype=bool),
            "z": np.array([[True, False]], dtype=bool),
            "h": np.array([[False, False]], dtype=bool),
            "s": np.array([[False, False]], dtype=bool),
            "sdg": np.array([[True, False]], dtype=bool),
            "sinv": np.array([[True, False]], dtype=bool),
            "v": np.array([[False, False]], dtype=bool),
            "w": np.array([[False, False]], dtype=bool),
        }

        target_stabilizer = {
            "i": "+Z",
            "id": "+Z",
            "iden": "+Z",
            "x": "-Z",
            "y": "-Z",
            "z": "+Z",
            "h": "+X",
            "s": "+Z",
            "sdg": "+Z",
            "sinv": "+Z",
            "v": "+X",
            "w": "+Y",
        }

        target_destabilizer = {
            "i": "+X",
            "id": "+X",
            "iden": "+X",
            "x": "+X",
            "y": "-X",
            "z": "-X",
            "h": "+Z",
            "s": "+Y",
            "sdg": "-Y",
            "sinv": "-Y",
            "v": "+Y",
            "w": "+Z",
        }

        for gate_name in ("i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "v", "w"):
            with self.subTest(msg="append gate %s" % gate_name):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff = _append_circuit(cliff, gate_name, [0])
                value_table = cliff.table._array
                value_phase = cliff.table._phase
                value_stabilizer = cliff.stabilizer.to_labels()
                value_destabilizer = cliff.destabilizer.to_labels()
                self.assertTrue(np.all(np.array(value_table == target_table[gate_name])))
                self.assertTrue(np.all(np.array(value_phase == target_phase[gate_name])))
                self.assertTrue(
                    np.all(np.array(value_stabilizer == [target_stabilizer[gate_name]]))
                )
                self.assertTrue(
                    np.all(np.array(value_destabilizer == [target_destabilizer[gate_name]]))
                )

    def test_1_qubit_identity_relations(self):
        """Tests identity relations for 1-qubit gates"""

        for gate_name in ("x", "y", "z", "h"):
            with self.subTest(msg="identity for gate %s" % gate_name):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_circuit(cliff, gate_name, [0])
                cliff = _append_circuit(cliff, gate_name, [0])
                self.assertEqual(cliff, cliff1)

        gates = ["s", "s", "v"]
        inv_gates = ["sdg", "sinv", "w"]

        for gate_name, inv_gate in zip(gates, inv_gates):
            with self.subTest(msg="identity for gate %s" % gate_name):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_circuit(cliff, gate_name, [0])
                cliff = _append_circuit(cliff, inv_gate, [0])
                self.assertEqual(cliff, cliff1)

    def test_1_qubit_mult_relations(self):
        """Tests multiplicity relations for 1-qubit gates"""

        rels = [
            "x * y = z",
            "x * z = y",
            "y * z = x",
            "s * s = z",
            "sdg * sdg = z",
            "sinv * sinv = z",
            "sdg * h = v",
            "h * s = w",
        ]

        for rel in rels:
            with self.subTest(msg="relation %s" % rel):
                split_rel = rel.split()
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_circuit(cliff, split_rel[0], [0])
                cliff = _append_circuit(cliff, split_rel[2], [0])
                cliff1 = _append_circuit(cliff1, split_rel[4], [0])
                self.assertEqual(cliff, cliff1)

    def test_1_qubit_conj_relations(self):
        """Tests conjugation relations for 1-qubit gates"""

        rels = [
            "h * x * h = z",
            "h * y * h = y",
            "s * x * sdg = y",
            "w * x * v = y",
            "w * y * v = z",
            "w * z * v = x",
        ]

        for rel in rels:
            with self.subTest(msg="relation %s" % rel):
                split_rel = rel.split()
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_circuit(cliff, split_rel[0], [0])
                cliff = _append_circuit(cliff, split_rel[2], [0])
                cliff = _append_circuit(cliff, split_rel[4], [0])
                cliff1 = _append_circuit(cliff1, split_rel[6], [0])
                self.assertEqual(cliff, cliff1)

    @combine(gate_name=("cx", "cz", "swap"), qubits=([0, 1], [1, 0]))
    def test_append_2_qubit_gate(self, gate_name, qubits):
        """Tests for append of 2-qubit gate {gate_name} {qubits}."""

        targets_cliffords = {
            "cx [0, 1]": Clifford(
                [
                    [True, True, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, True, True],
                ]
            ),
            "cx [1, 0]": Clifford(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, False, True],
                ]
            ),
            "cz [0, 1]": Clifford(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ),
            "cz [1, 0]": Clifford(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ),
            "swap [0, 1]": Clifford(
                [
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                ]
            ),
            "swap [1, 0]": Clifford(
                [
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                ]
            ),
        }

        gate_qubits = gate_name + " " + str(qubits)
        cliff = _append_circuit(Clifford(np.eye(4)), gate_name, qubits)
        target = targets_cliffords[gate_qubits]
        self.assertEqual(target, cliff)

    def test_2_qubit_identity_relations(self):
        """Tests identity relations for 2-qubit gates"""

        for gate_name in ("cx", "cz", "swap"):
            for qubits in ([0, 1], [1, 0]):
                with self.subTest(msg=f"append gate {gate_name} {qubits}"):
                    cliff = Clifford(np.eye(4))
                    cliff1 = cliff.copy()
                    cliff = _append_circuit(cliff, gate_name, qubits)
                    cliff = _append_circuit(cliff, gate_name, qubits)
                    self.assertEqual(cliff, cliff1)

    def test_2_qubit_relations(self):
        """Tests relations for 2-qubit gates"""

        with self.subTest(msg="relation between cx, h and cz"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_circuit(cliff, "h", [1])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "h", [1])
            cliff = _append_circuit(cliff, "cz", [0, 1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and swap"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "cx", [1, 0])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "swap", [0, 1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and x"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "x", [0])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "x", [0])
            cliff = _append_circuit(cliff, "x", [1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and z"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "z", [1])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "z", [0])
            cliff = _append_circuit(cliff, "z", [1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and s"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_circuit(cliff, "cx", [1, 0])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "s", [1])
            cliff = _append_circuit(cliff, "cx", [0, 1])
            cliff = _append_circuit(cliff, "cx", [1, 0])
            cliff = _append_circuit(cliff, "sdg", [0])
            self.assertEqual(cliff, cliff1)


@ddt
class TestCliffordSynthesis(QiskitTestCase):
    """Test Clifford synthesis methods."""

    def _cliffords_1q(self):
        clifford_dicts = [
            {"stabilizer": ["+Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-Z"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Z"]},
            {"stabilizer": ["+X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["+X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["+Y"]},
            {"stabilizer": ["-X"], "destabilizer": ["-Y"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+X"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-X"]},
            {"stabilizer": ["+Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["+Y"], "destabilizer": ["-Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["+Z"]},
            {"stabilizer": ["-Y"], "destabilizer": ["-Z"]},
        ]
        return [Clifford.from_dict(i) for i in clifford_dicts]

    def test_decompose_1q(self):
        """Test synthesis for all 1-qubit Cliffords"""
        for cliff in self._cliffords_1q():
            with self.subTest(msg=f"Test circuit {cliff}"):
                target = cliff
                value = Clifford(cliff.to_circuit())
                self.assertEqual(target, value)

    @combine(num_qubits=[2, 3])
    def test_decompose_2q_bm(self, num_qubits):
        """Test B&M synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for _ in range(samples):
            circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = Clifford(circ)
            value = Clifford(decompose_clifford_bm(target))
            self.assertEqual(value, target)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_decompose_2q_ag(self, num_qubits):
        """Test A&G synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for _ in range(samples):
            circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = Clifford(circ)
            value = Clifford(decompose_clifford_ag(target))
            self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_decompose_2q_greedy(self, num_qubits):
        """Test greedy synthesis for set of {num_qubits}-qubit Cliffords"""
        rng = np.random.default_rng(1234)
        samples = 50
        for _ in range(samples):
            circ = random_clifford_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = Clifford(circ)
            value = Clifford(decompose_clifford_greedy(target))
            self.assertEqual(value, target)


@ddt
class TestCliffordDecomposition(QiskitTestCase):
    """Test Clifford decompositions."""

    @combine(
        gates=[
            ["h", "s"],
            ["h", "s", "i", "x", "y", "z"],
            ["h", "s", "sdg"],
            ["h", "s", "v"],
            ["h", "s", "w"],
            ["h", "s", "sdg", "i", "x", "y", "z", "v", "w"],
        ]
    )
    def test_to_operator_1qubit_gates(self, gates):
        """Test 1-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 10
        seed = 100
        for i in range(samples):
            circ = random_clifford_circuit(1, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(
        gates=[
            ["cx"],
            ["cz"],
            ["swap"],
            ["cx", "cz"],
            ["cx", "swap"],
            ["cz", "swap"],
            ["cx", "cz", "swap"],
        ]
    )
    def test_to_operator_2qubit_gates(self, gates):
        """Test 2-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 10
        seed = 200
        for i in range(samples):
            circ = random_clifford_circuit(2, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(
        gates=[["h", "s", "cx"], ["h", "s", "cz"], ["h", "s", "swap"], "all"], num_qubits=[2, 3, 4]
    )
    def test_to_operator_nqubit_gates(self, gates, num_qubits):
        """Test {num_qubits}-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 20
        seed = 300
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_to_matrix(self, num_qubits):
        """Test to_matrix method"""
        samples = 10
        num_gates = 10
        seed = 333
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            mat = Clifford(circ).to_matrix()
            self.assertIsInstance(mat, np.ndarray)
            self.assertEqual(mat.shape, 2 * (2 ** num_qubits,))
            value = Operator(mat)
            target = Operator(circ)
            self.assertTrue(value.equiv(target))

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_to_circuit(self, num_qubits):
        """Test to_circuit method"""
        samples = 10
        num_gates = 10
        seed = 700
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            target = Clifford(circ)
            decomp = target.to_circuit()
            self.assertIsInstance(decomp, QuantumCircuit)
            self.assertEqual(decomp.num_qubits, circ.num_qubits)
            # Convert back to clifford and check it is the same
            self.assertEqual(Clifford(decomp), target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_to_instruction(self, num_qubits):
        """Test to_instruction method"""
        samples = 10
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            target = Clifford(circ)
            decomp = target.to_instruction()
            self.assertIsInstance(decomp, Gate)
            self.assertEqual(decomp.num_qubits, circ.num_qubits)
            # Convert back to clifford and check it is the same
            self.assertEqual(Clifford(decomp), target)


@ddt
class TestCliffordOperators(QiskitTestCase):
    """Test Clifford operator class methods."""

    @combine(num_qubits=[1, 2, 3])
    def test_is_unitary(self, num_qubits):
        """Test is_unitary method"""
        samples = 10
        num_gates = 10
        seed = 700
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).is_unitary()
            self.assertTrue(value)
        # tests a false clifford
        cliff = Clifford([[0, 0], [0, 1]], validate=False)
        value = cliff.is_unitary()
        self.assertFalse(value)

    @combine(num_qubits=[1, 2, 3])
    def test_conjugate(self, num_qubits):
        """Test conjugate method"""
        samples = 10
        num_gates = 10
        seed = 400
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).conjugate().to_operator()
            target = Operator(circ).conjugate()
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_transpose(self, num_qubits):
        """Test transpose method"""
        samples = 10
        num_gates = 1
        seed = 500
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).transpose().to_operator()
            target = Operator(circ).transpose()
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_compose_method(self, num_qubits):
        """Test compose method"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.compose(cliff2)
            target = Clifford(circ1.compose(circ2))
            self.assertEqual(target, value)

    @combine(num_qubits=[1, 2, 3])
    def test_dot_method(self, num_qubits):
        """Test dot method"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.dot(cliff2)
            target = Clifford(circ2.compose(circ1))
            self.assertEqual(target, value)

    @combine(num_qubits_1=[1, 2, 3], num_qubits_2=[1, 2, 3])
    def test_tensor_method(self, num_qubits_1, num_qubits_2):
        """Test tensor method"""
        samples = 5
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.tensor(cliff2)
            circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
            circ.append(circ2, range(num_qubits_2))
            circ.append(circ1, range(num_qubits_2, num_qubits_1 + num_qubits_2))
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[1, 2, 3], num_qubits_2=[1, 2, 3])
    def test_expand_method(self, num_qubits_1, num_qubits_2):
        """Test expand method"""
        samples = 5
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.expand(cliff2)
            circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
            circ.append(circ1, range(num_qubits_1))
            circ.append(circ2, range(num_qubits_1, num_qubits_1 + num_qubits_2))
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_compose_subsystem(self, num_qubits_1, num_qubits_2):
        """Test compose method of subsystems"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            circ = circ1.copy()
            circ.append(circ2.to_instruction(), qargs)
            value = Clifford(circ1).compose(Clifford(circ2), qargs)
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_dot_subsystem(self, num_qubits_1, num_qubits_2):
        """Test dot method of subsystems"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            circ = QuantumCircuit(num_qubits_1)
            circ.append(circ2.to_instruction(), qargs)
            circ.append(circ1.to_instruction(), range(num_qubits_1))
            value = Clifford(circ1).dot(Clifford(circ2), qargs)
            target = Clifford(circ)
            self.assertEqual(target, value)

    def test_to_dict(self):
        """Test to_dict method"""

        with self.subTest(msg="Identity"):
            cliff = Clifford(np.eye(8))
            value = cliff.to_dict()

            keys_value = set(value.keys())
            keys_target = {"destabilizer", "stabilizer"}
            self.assertEqual(keys_value, keys_target)

            stabilizer_value = set(value["stabilizer"])
            stabilizer_target = {"+IIIZ", "+IIZI", "+IZII", "+ZIII"}
            self.assertEqual(stabilizer_value, stabilizer_target)

            destabilizer_value = set(value["destabilizer"])
            destabilizer_target = {"+IIIX", "+IIXI", "+IXII", "+XIII"}
            self.assertEqual(destabilizer_value, destabilizer_target)

        with self.subTest(msg="bell"):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            cliff = Clifford(qc)
            value = cliff.to_dict()

            keys_value = set(value.keys())
            keys_target = {"destabilizer", "stabilizer"}
            self.assertEqual(keys_value, keys_target)

            stabilizer_value = set(value["stabilizer"])
            stabilizer_target = {"+XX", "+ZZ"}
            self.assertEqual(stabilizer_value, stabilizer_target)

            destabilizer_value = set(value["destabilizer"])
            destabilizer_target = {"+IZ", "+XI"}
            self.assertEqual(destabilizer_value, destabilizer_target)

    def test_from_dict(self):
        """Test from_dict method"""

        with self.subTest(msg="test raises not unitary"):
            cliff_dict = {"stabilizer": ["+XX", "+ZZ"], "destabilizer": ["+IZ", "+IY"]}
            self.assertRaises(QiskitError, Clifford.from_dict, cliff_dict)

        with self.subTest(msg="test raises wrong shape"):
            cliff_dict = {
                "stabilizer": ["+XX", "+ZZ", "+YY"],
                "destabilizer": ["+IZ", "+XI", "+IY"],
            }
            self.assertRaises(QiskitError, Clifford.from_dict, cliff_dict)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_dict_round_trip(self, num_qubits):
        """Test round trip conversion to and from dict"""
        num_gates = 10
        seed = 655
        gates = "all"
        circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + num_qubits)
        target = Clifford(circ)
        value = Clifford.from_dict(target.to_dict())
        self.assertEqual(value, target)

    def test_from_label(self):
        """Test from_label method"""
        label = "IXYZHS"
        CI = Clifford(IGate())
        CX = Clifford(XGate())
        CY = Clifford(YGate())
        CZ = Clifford(ZGate())
        CH = Clifford(HGate())
        CS = Clifford(SGate())
        target = CI.tensor(CX).tensor(CY).tensor(CZ).tensor(CH).tensor(CS)
        self.assertEqual(Clifford.from_label(label), target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_instruction_name(self, num_qubits):
        """Test to verify the correct clifford name is maintained
        after converting to instruction"""
        clifford = random_clifford(num_qubits, seed=777)
        self.assertEqual(clifford.to_instruction().name, str(clifford))


if __name__ == "__main__":
    unittest.main()
