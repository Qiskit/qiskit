# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependency import DAGDependency


def dag_to_dagdependency(dag):
    """Build a ``DAGDependency`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """

    dagdependency = DAGDependency()
    DAGDependency.name = dag.name

    qregs = list(dag.qregs.values())
    cregs = list(dag.cregs.values())

    for register in qregs:
        dagdependency.add_qreg(register)

    for register in cregs:
        dagdependency.add_creg(register)

    for node in dag.topological_op_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        inst.condition = node.condition
        dagdependency.add_op_node(inst, node.qargs, node.cargs)

    dagdependency._add_successors()

    return dagdependency
