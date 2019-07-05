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

"""Helper function for converting a circuit to a dag"""

from qiskit.dagcircuit.dagcircuit import DAGCircuit


def circuit_to_dag(circuit):
    """Build a ``DAGCircuit`` object from a ``QuantumCircuit``.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    for register in circuit.qregs:
        dagcircuit.add_qreg(register)
    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    for instruction, qargs, cargs in circuit.data:
        dagcircuit.apply_operation_back(instruction.copy(), qargs, cargs, instruction.control)

    return dagcircuit
