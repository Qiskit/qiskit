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

from typing import Type
from fnmatch import fnmatch
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag


class Decompose(TransformationPass):
    """Expand a gate in a circuit using its decomposition rules."""

    def __init__(self, gates_to_decompose: list = None):
        """Decompose initializer.

        Args:
            gate: gate to decompose.
            gates_to_decompose (list(str)): optional subset of gates to be decomposed,
            identified by gate name. Defaults to all gates.

        """
        super().__init__()
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
            haslabel = False

            if hasattr(node.op, 'label') and node.op.label is not None:
                haslabel = True

            if (
                self.gates is None or
                # check labels and label wildcards first
                (haslabel and (node.op.label in self.gates or
                any(fnmatch(node.op.label, p) for p in self.gates))) or
                # then check names and name wildcards
                (not haslabel and (node.name in self.gates or
                any(fnmatch(node.name, p) for p in self.gates)))
            ):
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
