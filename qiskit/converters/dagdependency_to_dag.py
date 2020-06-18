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

"""Helper function for converting a dag dependency to a dag circuit"""
from qiskit.dagcircuit.dagcircuit import DAGCircuit


def dagdependency_to_dag(dagdependency):
    """Build a ``DAGCircuit`` object from a ``DAGDependency``.

    Args:
        dag dependency (DAGDependency): the input dag.

    Return:
        DAGCircuit: the DAG representing the input circuit.
    """

    dagcircuit = DAGCircuit()
    dagcircuit.name = dagdependency.name

    qregs = list(dagdependency.qregs.values())
    cregs = list(dagdependency.cregs.values())

    for register in qregs:
        dagcircuit.add_qreg(register)

    for register in cregs:
        dagcircuit.add_creg(register)

    for node in dagdependency.get_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        inst.condition = node.condition

        dagcircuit.apply_operation_back(inst, node.qargs, node.cargs, inst.condition)

    return dagcircuit
