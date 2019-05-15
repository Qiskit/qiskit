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

"""Pass for unrolling a circuit to a given basis."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit import Parameter


class Unroller(TransformationPass):
    """
    Unroll (expand) non-basis, non-opaque instructions recursively
    to a desired basis, using decomposition rules defined for each instruction.
    """

    def __init__(self, basis):
        """
        Args:
            basis (list[str]): Target basis names to unroll to, e.g. `['u3', 'cx']` .
        """
        super().__init__()
        self.basis = basis

    def run(self, dag):
        """Expand all op nodes to the given basis.

        Args:
            dag(DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """
        # Walk through the DAG and expand each non-basis node
        for node in dag.op_nodes():
            basic_insts = ['measure', 'reset', 'barrier', 'snapshot']
            if node.name in basic_insts:
                # TODO: this is legacy behavior.Basis_insts should be removed that these
                #  instructions should be part of the device-reported basis. Currently, no
                #  backend reports "measure", for example.
                continue
            if node.name in self.basis:  # If already a base, ignore.
                continue

            # TODO: allow choosing other possible decompositions
            try:
                rule = node.op.definition
            except TypeError as err:
                if any(isinstance(p, Parameter) for p in node.op.params):
                    raise QiskitError('Unrolling gates parameterized by expressions '
                                      'is currently unsupported.')
                raise QiskitError('Error decomposing node {}: {}'.format(node.name, err))

            if not rule:
                raise QiskitError("Cannot unroll the circuit to the given basis, %s. "
                                  "No rule to expand instruction %s." %
                                  (str(self.basis), node.op.name))

            # hacky way to build a dag on the same register as the rule is defined
            # TODO: need anonymous rules to address wires by index
            decomposition = DAGCircuit()
            decomposition.add_qreg(rule[0][1][0][0])
            for inst in rule:
                decomposition.apply_operation_back(*inst)

            unrolled_dag = self.run(decomposition)  # recursively unroll ops
            dag.substitute_node_with_dag(node, unrolled_dag)
        return dag
