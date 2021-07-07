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

"""Expand a gate in a circuit using its decomposition rules."""
from typing import Type, Union, List, Optional
from fnmatch import fnmatch

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.circuit.gate import Gate
from qiskit.utils.deprecation import deprecate_arguments


class Decompose(TransformationPass):
    """Expand a gate in a circuit using its decomposition rules."""

    @deprecate_arguments({"gate": "gates_to_decompose"})
    def __init__(
        self,
        gate: Optional[Type[Gate]] = None,
        gates_to_decompose: Optional[Union[Type[Gate], List[Type[Gate]], List[str], str]] = None,
    ) -> None:
        """Decompose initializer.

        Args:
            gate: DEPRECATED gate to decompose.
            gates_to_decompose: optional subset of gates to be decomposed,
                identified by gate label, name or type. Defaults to all gates.
        """
        super().__init__()

        if not isinstance(gates_to_decompose, list):
            gates_to_decompose = [gates_to_decompose]

        self.gate = gate
        self.gates = gates_to_decompose

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Decompose pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            output dag where ``gate`` was expanded.
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.op_nodes():
            if self.__should_decompose(node):
                if not node.op.definition:
                    continue
                # TODO: allow choosing among multiple decomposition rules
                rule = node.op.definition.data
                if len(rule) == 1 and len(node.qargs) == len(rule[0][1]) == 1:
                    if node.op.definition.global_phase:
                        dag.global_phase += node.op.definition.global_phase
                    dag.substitute_node(node, rule[0][0], inplace=True)
                else:
                    decomposition = circuit_to_dag(node.op.definition)
                    dag.substitute_node_with_dag(node, decomposition)

        return dag

    def __should_decompose(self, node) -> bool:
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose)."""
        has_label = False
        strings_list = [s for s in self.gates if isinstance(s, str)]
        gate_type_list = [g for g in self.gates if isinstance(g, type)]

        if hasattr(node.op, "label") and node.op.label is not None:
            has_label = True

        if self.gates == [None] and self.gate is None:  # check if no gates given
            return True
        elif has_label and (  # check if label or label wildcard is given
            node.op.label in self.gates or any(fnmatch(node.op.label, p) for p in strings_list)
        ):
            return True
        elif not has_label and (  # check if name or name wildcard is given
            node.name in self.gates or any(fnmatch(node.name, p) for p in strings_list)
        ):
            return True
        elif not has_label and (  # check if Gate type given
            any(isinstance(node.op, op) for op in gate_type_list)
            or (self.gate is not None and isinstance(node.op, self.gate))
        ):
            return True
        else:
            return False
