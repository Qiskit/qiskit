# -*- coding: utf-8 -*-

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

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit


class Decompose(TransformationPass):
    """Expand a gate in a circuit using its decomposition rules."""

    def __init__(self, gate=None):
        """Decompose initializer.

        Args:
            gate (qiskit.circuit.gate.Gate): Gate to decompose.
        """
        super().__init__()
        self.gate = gate

    def run(self, dag):
        """Run the Decompose pass on `dag`.

        Args:
            dag(DAGCircuit): input dag

        Returns:
            DAGCircuit: output dag where gate was expanded.
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.op_nodes(self.gate):
            # opaque or built-in gates are not decomposable
            if not node.op.definition:
                continue
            # TODO: allow choosing among multiple decomposition rules
            rule = node.op.definition

            if len(rule) == 1 and len(node.qargs) == len(rule[0][1]):
                dag.substitute_node(node, rule[0][0], inplace=True)
            else:
                # hacky way to build a dag on the same register as the rule is defined
                # TODO: need anonymous rules to address wires by index
                decomposition = DAGCircuit()
                qregs = {qb.register for inst in rule for qb in inst[1]}
                cregs = {cb.register for inst in rule for cb in inst[2]}
                for qreg in qregs:
                    decomposition.add_qreg(qreg)
                for creg in cregs:
                    decomposition.add_creg(creg)
                for inst in rule:
                    decomposition.apply_operation_back(*inst)
                dag.substitute_node_with_dag(node, decomposition)
        return dag
