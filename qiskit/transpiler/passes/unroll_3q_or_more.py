# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for decomposing 3q (or more) gates into 2q or 1q gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError


class Unroll3qOrMore(TransformationPass):
    """
    Recursively expands 3+ qubit gates until the circuit only contains
    1 qubit and 2qubit gates.
    """

    def run(self, dag):
        """Expand 3+ qubit gates using their decomposition rules.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag with maximum node degrees of 2
        Raises:
            QiskitError: if a 3q+ gate is not decomposable
        """
        for node in dag.threeQ_or_more_gates():
            # TODO: allow choosing other possible decompositions
            rule = node.op.definition
            if not rule:
                raise QiskitError("Cannot unroll all 3q or more gates. "
                                  "No rule to expand instruction %s." %
                                  node.op.name)

            # hacky way to build a dag on the same register as the rule is defined
            # TODO: need anonymous rules to address wires by index
            decomposition = DAGCircuit()
            decomposition.add_qreg(rule[0][1][0][0])
            for inst in rule:
                decomposition.apply_operation_back(*inst)
            decomposition = self.run(decomposition)  # recursively unroll
            dag.substitute_node_with_dag(node, decomposition)
        return dag
