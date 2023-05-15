# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if all gates in the DAGCircuit are in the specified basis gates."""

from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass


class GatesInBasis(AnalysisPass):
    """Check if all gates in a DAG are in a given set of gates"""

    def __init__(self, basis_gates=None, target=None):
        """Initialize the GatesInBasis pass.

        Args:
            basis_gates (list): The list of strings representing the set of basis gates.
            target (Target): The target representing the backend. If specified
                this will be used instead of the ``basis_gates`` parameter
        """
        super().__init__()
        self._basis_gates = None
        if basis_gates is not None:
            self._basis_gates = set(basis_gates).union(
                {"measure", "reset", "barrier", "snapshot", "delay"}
            )
        self._target = target

    def run(self, dag):
        """Run the GatesInBasis pass on `dag`."""
        if self._basis_gates is None and self._target is None:
            self.property_set["all_gates_in_basis"] = True
            return
        gates_out_of_basis = False
        if self._target is not None:

            def _visit_target(dag, wire_map):
                for gate in dag.op_nodes():
                    # Barrier is universal and supported by all backends
                    if gate.name == "barrier":
                        continue
                    if not self._target.instruction_supported(
                        gate.name, tuple(wire_map[bit] for bit in gate.qargs)
                    ):
                        return True
                    # Control-flow ops still need to be supported, so don't skip them in the
                    # previous checks.
                    if isinstance(gate.op, ControlFlowOp):
                        for block in gate.op.blocks:
                            inner_wire_map = {
                                inner: wire_map[outer]
                                for outer, inner in zip(gate.qargs, block.qubits)
                            }
                            if _visit_target(circuit_to_dag(block), inner_wire_map):
                                return True
                return False

            qubit_map = {qubit: index for index, qubit in enumerate(dag.qubits)}
            gates_out_of_basis = _visit_target(dag, qubit_map)
        else:
            for gate in dag.count_ops(recurse=True):
                if gate not in self._basis_gates:
                    gates_out_of_basis = True
                    break
        self.property_set["all_gates_in_basis"] = not gates_out_of_basis
