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

"""Helper function for converting a dag dependency to a circuit"""
from qiskit.circuit import QuantumCircuit


def dagdependency_to_circuit(dagdependency):
    """Build a ``QuantumCircuit`` object from a ``DAGDependency``.

    Args:
        dagdependency (DAGDependency): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag dependency.
    """

    name = dagdependency.name or None
    circuit = QuantumCircuit(*dagdependency.qregs.values(), *dagdependency.cregs.values(),
                             name=name)

    for node in dagdependency.get_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        inst.condition = node.condition
        circuit._append(inst, node.qargs, node.cargs)

    return circuit
