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

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import PBCTransformation
from qiskit.quantum_info import Operator
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
    IGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    HGate,
    CXGate,
    CYGate,
    CZGate,
    CHGate,
    CSGate,
    CSdgGate,
    CSXGate,
    SwapGate,
    iSwapGate,
    DCXGate,
    ECRGate,
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
        global_phase=[0, 1.0, -3.0],
    )
    def test_single_param_gates_transpiled(self, gate, angle, global_phase):
        """Test that standard 1-qubit and 2-qubit single parametric gates are translated into
        Pauli product rotatations correctly."""
        num_qubits = gate(angle).num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate(angle), range(num_qubits))
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))

    @combine(
        gate=[
            IGate,
            XGate,
            YGate,
            ZGate,
            SGate,
            SdgGate,
            TGate,
            TdgGate,
            HGate,
            CXGate,
            CYGate,
            CZGate,
            CHGate,
            CSGate,
            CSdgGate,
            CSXGate,
            SwapGate,
            iSwapGate,
            DCXGate,
            ECRGate,
        ],
        global_phase=[0, 1.0, -3.0],
    )
    def test_non_params_gates_transpiled(self, gate, global_phase):
        """Test that standard 1-qubit and 2-qubit non-parametric gates are translated into
        Pauli product rotatations correctly."""
        num_qubits = gate().num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate(), range(num_qubits))
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))
