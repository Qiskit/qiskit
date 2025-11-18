# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Discretize Rotations optimization pass"""

from ddt import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit
from qiskit.transpiler.passes import DiscretizeRotations
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.circuit.library import RXGate, RYGate, RZGate
from test import combine, QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestDiscretizeRotations(QiskitTestCase):
    """Test the Discretize Rotations optimization pass."""

    @combine(multiple=[*range(0, 16), 23, 42, -5, -8, -17, -22, -35], gate=[RXGate, RYGate, RZGate])
    def test_rotation_gates_transpiled(self, multiple, gate):
        """Test that rotations gates are translated into Clifford+T+Tdg correctly."""
        qc = QuantumCircuit(1)
        angle = np.pi / 4 * multiple
        qc.append(gate(angle), [0])
        qct = DiscretizeRotations()(qc)
        ops = qct.count_ops()
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(qct.size(), 4)
        self.assertLessEqual(set(ops.keys()), set(clifford_t_names))
        self.assertLessEqual(
            len(set(ops.keys()).intersection({"t", "tdg"})), 1
        )  # at most one t/tdg gate
        if multiple % 2 == 0:  # only clifford gates
            self.assertLessEqual(qct.size(), 3)
            self.assertLessEqual(set(ops.keys()), set(get_clifford_gate_names()))
            self.assertEqual(ops.get("t", 0) + ops.get("tdg", 0), 0)
        else:
            self.assertEqual(ops.get("t", 0) + ops.get("tdg", 0), 1)

    @combine(multiple=[*range(0, 16)], eps=[0.001, -0.001], gate=[RXGate, RYGate, RZGate])
    def test_rotation_gates_do_not_change(self, multiple, eps, gate):
        """Test that the transpiler pass does not change the gates for angles that are
        not 2*pi/8 rotations."""
        qc = QuantumCircuit(1)
        angle = np.pi / 4 * multiple + eps
        qc.append(gate(angle), [0])
        qct = DiscretizeRotations()(qc)
        self.assertEqual(qc, qct)

    def test_random_clifford_t_rotation_circuit(self):
        """Test that pseudo-random Clifford+T+Tdg+rotation circuits are transpiled correctly."""
        num_qubits = 5
        num_gates = 10
        qc = QuantumCircuit(num_qubits)
        for idx in range(num_qubits):
            cliff_qc = random_clifford_circuit(num_qubits, num_gates, seed=idx)
            qc.compose(cliff_qc, inplace=True)
            qc.t(idx)
            qc.tdg((idx + 1) % num_qubits)
            qc.rx(np.pi / 4 * (idx + num_qubits + 2), (idx + 2) % num_qubits)
            qc.ry(np.pi / 4 * (idx + num_qubits + 3), (idx + 3) % num_qubits)
            qc.rz(np.pi / 4 * (idx + num_qubits + 4), (idx + 4) % num_qubits)

        qct = DiscretizeRotations()(qc)
        clifford_t_names = get_clifford_gate_names() + ["t"] + ["tdg"]
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(set(qct.count_ops().keys()), set(clifford_t_names))

    def test_non_rotation_circuit(self):
        """Test that pseudo-random circuits including rotation gates, other gates and barriers
        are transpiled correctly."""
        num_qubits = 5
        qc = QuantumCircuit(num_qubits)
        qc.t(0)
        qc.tdg(1)
        qc.rx(np.pi / 4 * (num_qubits + 2), 2)
        qc.ry(np.pi / 4 * (num_qubits + 3), 2)
        qc.rz(np.pi / 4 * (num_qubits + 4), 4)
        qc.u(0.12, 0.34, 0.56, 3)
        qc.cx(0, 1)
        qc.cz(1, 2)
        qc.rzz(0.1, 0, 1)
        qc.p(0.34, 4)
        qc.barrier()

        qct = DiscretizeRotations()(qc)
        gate_names = (
            get_clifford_gate_names()
            + ["t"]
            + ["tdg"]
            + ["u"]
            + ["p"]
            + ["rzz"]
            + ["cx"]
            + ["cz"]
            + ["rzz"]
            + ["barrier"]
        )
        self.assertEqual(Operator(qct), Operator(qc))
        self.assertLessEqual(set(qct.count_ops().keys()), set(gate_names))
