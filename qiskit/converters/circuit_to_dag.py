# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to a dag"""

import copy

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
        # Get arguments for classical control (if any)
        if instruction.control is None:
            control = None
        else:
            control = (instruction.control[0], instruction.control[1])

        instruction = copy.deepcopy(instruction)
        dagcircuit.apply_operation_back(instruction, qargs, cargs, control)

    return dagcircuit
