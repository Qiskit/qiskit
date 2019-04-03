# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a circuit to an instruction"""

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister


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

    def find_bit_position(bit):
        """find the index of a given bit (Register, int) within
        a flat ordered list of bits of the circuit
        """
        if isinstance(bit[0], QuantumRegister):
            ordered_regs = circuit.qregs
        else:
            ordered_regs = circuit.cregs
        reg_index = ordered_regs.index(bit[0])
        return sum([reg.size for reg in ordered_regs[:reg_index]]) + bit[1]

    definition = circuit.data.copy()

    if instruction.num_qubits > 0:
        q = QuantumRegister(instruction.num_qubits, 'q')
    if instruction.num_clbits > 0:
        c = ClassicalRegister(instruction.num_clbits, 'c')

    definition = list(map(lambda x:
                          (x[0],
                           list(map(lambda y: (q, find_bit_position(y)), x[1])),
                           list(map(lambda y: (c, find_bit_position(y)), x[2]))), definition))
    instruction.definition = definition

    return instruction
