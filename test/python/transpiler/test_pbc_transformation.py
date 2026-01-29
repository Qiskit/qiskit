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

"""Test PBCTransformation optimization pass"""

from ddt import ddt

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.passes import PBCTransformation
from qiskit.quantum_info import Operator
from qiskit.circuit.random import random_circuit
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
    UGate,
    U3Gate,
    U2Gate,
    RGate,
    CUGate,
    CU3Gate,
    XXPlusYYGate,
    XXMinusYYGate,
    CCXGate,
    C3XGate,
    C4XGate,
    MCXGate,
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

    @combine(
        gate=[
            UGate(0.12, -0.34, 0.56),
            U3Gate(0.13, -0.24, 0.65),
            RGate(0.12, -0.34),
            U2Gate(0.12, -0.34),
            CUGate(0.12, -0.34, 0.56, -0.78),
            CU3Gate(0.13, -0.24, 0.67),
            XXPlusYYGate(0.12, -0.34),
            XXMinusYYGate(0.12, -0.34),
        ],
        global_phase=[0, 1.0, -3.0],
    )
    def test_many_param_gates_transpiled(self, gate, global_phase):
        """Test that standard 1-qubit and 2-qubit gates with several paramteres are translated into
        Pauli product rotatations correctly."""
        num_qubits = gate.num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.global_phase = global_phase
        qc.append(gate, range(num_qubits))
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit(self):
        """Test that a pesudo-random circuit with 1-qubit and 2-qubit gates
        is tranlated into Pauli product rotatations correctly."""
        num_qubits = 5
        depth = 200
        seed = 1234
        qc = random_circuit(num_qubits=num_qubits, depth=depth, max_operands=2, seed=seed)
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit_measure_barrier(self):
        """Test that a pesudo-random circuit with 1-qubit and 2-qubit gates,
        measurements, delays, resets and barriers,
        is tranlated into Pauli product rotatations correctly."""
        num_qubits = 4
        depth = 10
        seed = 5678
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc1 = random_circuit(num_qubits=num_qubits, depth=depth, max_operands=2, seed=seed)
            qc.compose(qc1, inplace=True)
            qc.delay(i)
            qc.reset((i + 1) % num_qubits)
            qc.barrier()
        qc.measure_all()
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(
            ops_names, {"PauliEvolution", "pauli_product_measurement", "delay", "reset", "barrier"}
        )

    @combine(
        gate=[
            RXGate(Parameter("theta")),
            RYGate(Parameter("theta")),
            RZGate(Parameter("theta")),
            PhaseGate(Parameter("theta")),
            U1Gate(Parameter("theta")),
            RZZGate(Parameter("theta")),
            RXXGate(Parameter("theta")),
            RZXGate(Parameter("theta")),
            RYYGate(Parameter("theta")),
            CPhaseGate(Parameter("theta")),
            CU1Gate(Parameter("theta")),
            CRZGate(Parameter("theta")),
            CRXGate(Parameter("theta")),
            CRYGate(Parameter("theta")),
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
            U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
            RGate(Parameter("theta"), Parameter("phi")),
            U2Gate(Parameter("theta"), Parameter("phi")),
            CUGate(Parameter("theta"), Parameter("phi"), Parameter("lam"), Parameter("gamma")),
            CU3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
            XXPlusYYGate(Parameter("theta"), Parameter("phi")),
            XXMinusYYGate(Parameter("theta"), Parameter("phi")),
        ],
    )
    def test_parametrized_gates(self, gate):
        """Test that a circuit with 1-qubit and 2-qubit parametrized gates
        is tranlated into Pauli product rotatations correctly."""
        num_qubits = gate.num_qubits
        num_params = len(gate.params)
        qc = QuantumCircuit(num_qubits)
        qc.append(gate, range(num_qubits))
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        qc_bound = qc.assign_parameters([0.123] * num_params)
        qct_bound = qct.assign_parameters([0.123] * num_params)
        self.assertEqual(Operator(qct_bound), Operator(qc_bound))

    @combine(
        gate=[
            CCXGate(),
            C3XGate(),
            C4XGate(),
            MCXGate(5),
        ]
    )
    def test_unsupported_gates_raise_error(self, gate):
        """Test that unsupported gates raise a trasnpiler error."""
        num_qubits = gate.num_qubits
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(gate, range(num_qubits))
        qc.rzz(0.123, 0, 1)

        with self.assertRaises(TranspilerError):
            _ = PBCTransformation()(qc)
