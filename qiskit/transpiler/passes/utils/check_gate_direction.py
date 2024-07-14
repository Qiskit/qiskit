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

"""Check if the gates follow the right direction with respect to the coupling map."""

from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass


class CheckGateDirection(AnalysisPass):
    """Check if the two-qubit gates follow the right direction with
    respect to the coupling map.
    """

    def __init__(self, coupling_map, target=None):
        """CheckGateDirection initializer.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            target (Target): The backend target to use for this pass. If this is specified
                it will be used instead of the coupling map
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

    def _coupling_map_visit(self, dag, wire_map, edges=None):
        if edges is None:
            edges = self.coupling_map.get_edges()
        # Don't include directives to avoid things like barrier, which are assumed always supported.
        for node in dag.op_nodes(include_directives=False):
            if node.name in CONTROL_FLOW_OP_NAMES:
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }

                    if not self._coupling_map_visit(circuit_to_dag(block), inner_wire_map, edges):
                        return False
            elif (
                len(node.qargs) == 2
                and (wire_map[node.qargs[0]], wire_map[node.qargs[1]]) not in edges
            ):
                return False
        return True

    def _target_visit(self, dag, wire_map):
        # Don't include directives to avoid things like barrier, which are assumed always supported.
        for node in dag.op_nodes(include_directives=False):
            if node.name in CONTROL_FLOW_OP_NAMES:
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    if not self._target_visit(circuit_to_dag(block), inner_wire_map):
                        return False
            elif len(node.qargs) == 2 and not self.target.instruction_supported(
                node.name, (wire_map[node.qargs[0]], wire_map[node.qargs[1]])
            ):
                return False
        return True

    def run(self, dag):
        """Run the CheckGateDirection pass on `dag`.

        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to check.
        """
        wire_map = {bit: i for i, bit in enumerate(dag.qubits)}
        self.property_set["is_direction_mapped"] = (
            self._coupling_map_visit(dag, wire_map)
            if self.target is None
            else self._target_visit(dag, wire_map)
        )
