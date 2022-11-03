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

"""Translate parameterized gates only, and leave others as they are."""

from __future__ import annotations

from qiskit.circuit import Instruction, ParameterExpression, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.preset_passmanagers.common import generate_translation_passmanager


class TranslateParameterizedGates(TransformationPass):
    """Translate parameterized gates to a supported basis set."""

    def __init__(self, supported_gates: list[str]) -> None:
        """
        Args:
            supported_gates: A list of suppported basis gates specified as string.
        """
        super().__init__()
        self.supported_gates = supported_gates
        self.translator = generate_translation_passmanager(target=None, basis_gates=supported_gates)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the transpiler pass.

        Args:
            dag: The DAG circuit in which the parameterized gates should be unrolled.

        Returns:
            A DAG where the parameterized gates have been unrolled.

        Raises:
            ValueError: If the circuit cannot be unrolled.
        """
        for node in dag.op_nodes():
            # check whether it is parameterized and we need to decompose it
            if _is_parameterized(node.op) and (node.op.name not in self.supported_gates):
                # translate the instruction and replace the node
                translated = self.translator.run(_instruction_to_circuit(node.op))
                dag.substitute_node_with_dag(node, circuit_to_dag(translated))

        return dag


def _is_parameterized(op: Instruction) -> bool:
    return any(
        isinstance(param, ParameterExpression) and len(param.parameters) > 0 for param in op.params
    )


def _instruction_to_circuit(op: Instruction) -> QuantumCircuit:
    circuit = QuantumCircuit(op.num_qubits, op.num_clbits)
    circuit._append(op, circuit.qubits, circuit.clbits)
    return circuit
