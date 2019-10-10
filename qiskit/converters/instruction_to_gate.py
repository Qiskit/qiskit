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

"""
Convert Instruction object to Gate object.
"""
from qiskit.exceptions import QiskitError


def instruction_to_gate(instruction):
    """Attempt to convert instruction to gate.

    Args:
        instruction (Instruction): instruction to convert

    Returns:
        Gate: instruction cast to gate

    Raises:
        QiskitError: Conversion fails if element of definition can't be converted.
    """
    from qiskit.circuit.gate import Gate
    from qiskit.circuit.instruction import Instruction
    gate_spec_list = []
    if instruction.definition is not None:
        for instr_spec in instruction.definition:
            instr = instr_spec[0]
            if isinstance(instr, Gate):
                thisgate = instr
            elif isinstance(instr, Instruction):
                thisgate = instruction_to_gate(instr)
            else:
                raise QiskitError('One or more instructions in this instruction '
                                  'cannot be converted to a gate')
            gate_spec_list.append((thisgate, instr_spec[1], instr_spec[2]))
    else:
        raise QiskitError('This instruction ({}) cannot be converted to a gate'.format(
            instruction.name))
    gate = Gate(instruction.name, instruction.num_qubits, instruction.params)
    gate.definition = gate_spec_list
    return gate
