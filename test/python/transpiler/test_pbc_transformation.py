# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Substitute Pi/4-Rotations optimization pass"""

from ddt import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit
from qiskit.transpiler.passes import PBCTransformation
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.circuit.library import (
    RXGate,
    RYGate,
    RZGate,
    PhaseGate,
    U1Gate,
    RZZGate,
    RXXGate,
    RZXGate,
    RYYGate,
    CPhaseGate,
    CU1Gate,
    CRZGate,
    CRXGate,
    CRYGate,
)
from test import combine, QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPBCTransformation(QiskitTestCase):
    """Test the PBC Transformation optimization pass."""

    @combine(
        gate=[
            RXGate,
            RYGate,
            RZGate,
            PhaseGate,
            U1Gate,
            RZZGate,
            RXXGate,
            RZXGate,
            RYYGate,
            CPhaseGate,
            CU1Gate,
            CRZGate,
            CRXGate,
            CRYGate,
        ],
        angle=[0.1, -0.2],
        global_phase=[0],  # , 1.0, -3.0],
    )
    def test_rotation_gates_transpiled(self, gate, angle, global_phase):
        """Test that standard 1-qubit and 2-qubit rotation gates are translated into Pauli product rotatations correctly."""
        num_qubits = gate(angle).num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate(angle), range(num_qubits))
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))
