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

"""Test PBCTransformation pass"""

from ddt import ddt

from qiskit.circuit import QuantumCircuit, Parameter, Instruction
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.passes import PBCTransformation
from qiskit.quantum_info import Operator
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.library import (
    CCXGate,
    C3XGate,
    C4XGate,
    MCXGate,
    GlobalPhaseGate,
)
from test import combine, QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPBCTransformation(QiskitTestCase):
    """Test the PBC Transformation pass."""

    def setUp(self):
        super().setUp()
        self.standard_gates = get_standard_gate_name_mapping()

    @combine(angle=[0.12, -0.43], global_phase=[0, 1.0, -3.0])
    def test_standard_gates_transpiled(self, angle, global_phase):
        """Test that standard 1-qubit and 2-qubit gates are translated into
        Pauli product rotatations correctly."""
        for gate in self.standard_gates.values():
            if isinstance(gate, Instruction):
                continue  # we only test gates, not instructions like "Reset"
            num_qubits = gate.num_qubits
            if num_qubits in [1, 2]:
                params = [angle * (i + 1) for i in range(len(gate.params))]
                qc = QuantumCircuit(num_qubits)
                qc.global_phase = global_phase
                qc.append(gate.base_class(*params), range(num_qubits))
                qct = PBCTransformation()(qc)
                ops_names = set(qct.count_ops().keys())
                self.assertEqual(ops_names, {"PauliEvolution"})
                self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit(self):
        """Test that a pesudo-random circuit with 1-qubit and 2-qubit gates
        is translated into Pauli product rotations correctly."""
        num_qubits = 5
        depth = 200
        seed = 1234
        qc = random_circuit(num_qubits=num_qubits, depth=depth, max_operands=2, seed=seed)
        qct = PBCTransformation()(qc)
        ops_names = set(qct.count_ops().keys())
        self.assertEqual(ops_names, {"PauliEvolution"})
        self.assertEqual(Operator(qct), Operator(qc))

    def test_random_circuit_measure_barrier_delay_reset(self):
        """Test that a pesudo-random circuit with 1-qubit and 2-qubit gates,
        measurements, delays, resets and barriers,
        is translated into Pauli product rotations correctly."""
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

    def test_parametrized_gates(self):
        """Test that a circuit with 1-qubit and 2-qubit parametrized gates
        is translated into Pauli product rotations correctly."""
        symbols = [Parameter("theta"), Parameter("phi"), Parameter("lam"), Parameter("gamma")]
        for gate in self.standard_gates.values():
            if isinstance(gate, Instruction):
                continue  # we only test gates, not instructions like "Reset"
            num_qubits = gate.num_qubits
            num_params = len(gate.params)
            if num_qubits in [1, 2]:
                params = symbols[:num_params]
                qc = QuantumCircuit(num_qubits)
                qc.append(gate.base_class(*params), range(num_qubits))
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
            GlobalPhaseGate(0.1),
        ]
    )
    def test_unsupported_gates_raise_error(self, gate):
        """Test that unsupported gates raise a transpiler error."""
        num_qubits = gate.num_qubits
        qc = QuantumCircuit(num_qubits + 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(gate, range(num_qubits))
        qc.rzz(0.123, 0, 1)

        with self.assertRaises(TranspilerError):
            _ = PBCTransformation()(qc)
