# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

'''
Convert a QuantumCircuit object to the dag canonical form
'''
from qiskit.dagcircuit.dagcanonical import DAGcanonical

def circuit_to_canonical(circuit):
    """Build a ``DAGCanonical`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuits.

    Return:
        DAGcanonical: the DAG representing the input circuit as in the canonical form.
    """
    dagcircuit = DAGcanonical()
    dagcircuit.name = circuit.name

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for operation, qargs, cargs in circuit.data:
        if operation.name not in {"barrier", "snapshot", "measure", "reset", "copy"}:
            dagcircuit.add_node(operation, qargs, cargs)
            dagcircuit.add_edge()

    dagcircuit.add_successors()

    return dagcircuit
