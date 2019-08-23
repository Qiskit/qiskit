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
            qregs = {qb.register for inst in rule for qb in inst[1]}
            cregs = {cb.register for inst in rule for cb in inst[2]}
            for qreg in qregs:
                decomposition.add_qreg(qreg)
            for creg in cregs:
                decomposition.add_creg(creg)
            for inst in rule:
                decomposition.apply_operation_back(*inst)
            decomposition = self.run(decomposition)  # recursively unroll
            dag.substitute_node_with_dag(node, decomposition)
        return dag
