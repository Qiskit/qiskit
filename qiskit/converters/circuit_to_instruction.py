# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to an instruction"""

import copy

from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag

def circuit_to_instruction(circuit):
    """Build a ``Instruction`` object from a ``QuantumCircuit``.

    The instruction is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The instruction will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.

    Return:
        Instruction: an instruction equivalent to the action of the
            input circuit. Upon decomposition, this instruction will
            yield the components comprising the original circuit.
    """
    instruction = Instruction(name=circuit.name,
                              num_qubits=sum([qreg.size for qreg in circuit.qregs]),
                              num_clbits=sum([qreg.size for qreg in circuit.cregs]),
                              params=[])
    instruction.control = None
    instruction.circuit = None

    instruction._decompositions = [circuit_to_dag(circuit)]

    return instruction
