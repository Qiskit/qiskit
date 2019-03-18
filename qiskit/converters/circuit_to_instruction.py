# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to an instruction"""

from qiskit.circuit.instruction import Instruction


def circuit_to_instruction(circuit):
    """Build an ``Instruction`` object from a ``QuantumCircuit``.

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
                              num_clbits=sum([creg.size for creg in circuit.cregs]),
                              params=[])
    instruction.control = None

    instruction.definition = circuit.data.copy()

    return instruction
