# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a circuit to an instruction"""

from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister


def circuit_to_instruction(circuit, parameter_map=None):
    """Build an ``Instruction`` object from a ``QuantumCircuit``.

    The instruction is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The instruction will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the instruction.
           If None, existing circuit parameters will also parameterize the
           instruction.

    Raises:
        QiskitError: if parameter_map is not compatible with circuit

    Return:
        Instruction: an instruction equivalent to the action of the
            input circuit. Upon decomposition, this instruction will
            yield the components comprising the original circuit.
    """

    if parameter_map is None:
        parameter_map = {p: p for p in circuit.parameters}

    if parameter_map.keys() != circuit.parameters:
        raise QiskitError(('parameter_map should map all circuit parameters. '
                           'Circuit parameters: {}, parameter_map: {}').format(
                               circuit.parameters, parameter_map))

    instruction = Instruction(name=circuit.name,
                              num_qubits=sum([qreg.size for qreg in circuit.qregs]),
                              num_clbits=sum([creg.size for creg in circuit.cregs]),
                              params=sorted(parameter_map.values(), key=lambda p: p.name))
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

    target = circuit.copy()
    target._substitute_parameters(parameter_map)

    definition = target.data

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
