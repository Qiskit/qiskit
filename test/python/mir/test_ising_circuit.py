# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Any
from ddt import ddt
import unittest

from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit._accelerate import mir, mir_passes
from qiskit.passmanager import LoweringPassManager
from qiskit.transpiler import generate_preset_pbc_pass_manager
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from test import QiskitTestCase, combine
from test.python.mir.pass_adaptor import CircuitToDag, LegacyDagPass

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]
NODES = 1 + max(max(e) for e in EDGES)


def make_ising_circuit(steps=1, h=0.5):
    """Build a Trotterized transverse-field Ising circuit on a 2x2 grid of qubits."""
    J = 1.0  # Ising coupling
    dt = 0.1  # Trotter step size
    qubits = [Qubit() for _ in range(NODES)]
    clbits = [Clbit() for _ in range(NODES)]
    qc = QuantumCircuit(qubits, clbits)
    for _ in range(steps):
        # ZZ couplings
        for i, j in EDGES:
            pauli = Pauli("ZZ")
            gate = PauliEvolutionGate(pauli, time=-J * dt)
            qc.append(gate, [qubits[i], qubits[j]])
        # transverse X fields
        for i in range(NODES):
            pauli = Pauli("X")
            gate = PauliEvolutionGate(pauli, time=-h * dt)
            qc.append(gate, [qubits[i]])
    for i in range(NODES):
        qc.measure(qubits[i], clbits[i])
    return qc


@ddt
class TestIsingCircuitMIR(QiskitTestCase):
    """Unit tests for lowering Ising 2x2 circuit to MIR."""

    def setUp(self):
        super().setUp()
        self.pm = LoweringPassManager(
            [CircuitToDag()]
            + [
                LegacyDagPass(pass_)
                for pass_ in generate_preset_pbc_pass_manager().to_flow_controller().tasks
            ]
            + [mir_passes.DagToMirPass()]
        )

    @combine(steps=[1, 10, 30], h=[0.5, 0.8])
    def test_ising_circuit(self, steps, h):
        """Test lowering Ising circuit to MIR."""
        qc = make_ising_circuit(steps, h)
        mir_instance: Any = self.pm.run(qc)
        self.log.debug("steps=%s, h=%s, mir_instance=\n%s", steps, h, mir_instance)
        self.assertIsInstance(mir_instance, mir.Mir)
        self.assertEqual(mir_instance.num_qids(), NODES)
        num_instructions = steps * (len(EDGES) + NODES) + NODES
        self.assertEqual(mir_instance.num_instructions(), num_instructions)


if __name__ == "__main__":
    unittest.main()
