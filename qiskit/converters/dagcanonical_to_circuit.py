# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag canonical to a circuit"""
from qiskit.circuit import QuantumCircuit


def dagcanonical_to_circuit(dagcanonical):
    """Build a ``QuantumCircuit`` object from a ``DAGCanonical``.

    Args:
        dag (DAGCcanonical): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag.
    """

    name = dagcanonical.name or None
    circuit = QuantumCircuit(*dagcanonical.qregs.values(), *dagcanonical.cregs.values(), name=name)

    for node in list(dagcanonical.nodes()):
        node_op=node[1]
        # Get arguments for classical control (if any)
        inst = node_op.op.copy()
        inst.condition = node_op.condition
        circuit._append(inst, node_op.qargs, node_op.cargs)

    return circuit
