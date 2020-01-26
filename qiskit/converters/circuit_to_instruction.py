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

"""Helper function for converting a circuit to an instruction."""

from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
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

    Example:
        .. jupyter-execute::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.converters import circuit_to_instruction
            %matplotlib inline

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            circuit_to_instruction(circ)
    """

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(('parameter_map should map all circuit parameters. '
                           'Circuit parameters: {}, parameter_map: {}').format(
                               circuit.parameters, parameter_dict))

    instruction = Instruction(name=circuit.name,
                              num_qubits=sum([qreg.size for qreg in circuit.qregs]),
                              num_clbits=sum([creg.size for creg in circuit.cregs]),
                              params=sorted(parameter_dict.values(), key=lambda p: p.name))
    instruction.condition = None

    def find_bit_position(bit):
        """find the index of a given bit (Register, int) within
        a flat ordered list of bits of the circuit
        """
        if isinstance(bit, Qubit):
            ordered_regs = circuit.qregs
        else:
            ordered_regs = circuit.cregs
        reg_index = ordered_regs.index(bit.register)
        return sum([reg.size for reg in ordered_regs[:reg_index]]) + bit.index

    target = circuit.copy()
    target._substitute_parameters(parameter_dict)

    definition = target.data

    if instruction.num_qubits > 0:
        q = QuantumRegister(instruction.num_qubits, 'q')
    if instruction.num_clbits > 0:
        c = ClassicalRegister(instruction.num_clbits, 'c')

    definition = list(map(lambda x:
                          (x[0],
                           list(map(lambda y: q[find_bit_position(y)], x[1])),
                           list(map(lambda y: c[find_bit_position(y)], x[2]))), definition))
    instruction.definition = definition

    return instruction
