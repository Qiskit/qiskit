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

"""
Instruction collection.
"""
from qiskit.circuit.exceptions import CircuitError
from .instruction import Instruction
from .classicalregister import Clbit


class InstructionSet:
    """Instruction collection, and their contexts."""

    def __init__(self, circuit_cregs=None):
        """New collection of instructions.

        The context (qargs and cargs that each instruction is attached to)
        is also stored separately for each instruction.

        Args:
            circuit_cregs (list[ClassicalRegister]): Optional. List of cregs of the
                circuit to which the instruction is added. Default: `None`.
        """
        self.instructions = []
        self.qargs = []
        self.cargs = []
        self.circuit_cregs = circuit_cregs

    def __len__(self):
        """Return number of instructions in set"""
        return len(self.instructions)

    def __getitem__(self, i):
        """Return instruction at index"""
        return self.instructions[i]

    def add(self, gate, qargs, cargs):
        """Add an instruction and its context (where it is attached)."""
        if not isinstance(gate, Instruction):
            raise CircuitError("attempt to add non-Instruction" + " to InstructionSet")
        self.instructions.append(gate)
        self.qargs.append(qargs)
        self.cargs.append(cargs)

    def inverse(self):
        """Invert all instructions."""
        for index, instruction in enumerate(self.instructions):
            self.instructions[index] = instruction.inverse()
        return self

    def c_if(self, classical, val):
        """Add condition on classical register to all instructions."""
        if self.circuit_cregs:
            circuit_clbits = [bit for creg in self.circuit_cregs for bit in creg]
        if isinstance(classical, (Clbit, int)):
            if isinstance(classical, int):
                classical = circuit_clbits[classical]
            for creg in self.circuit_cregs:
                if creg[:] == [classical]:
                    classical = creg
                    break
        for gate in self.instructions:
            gate.c_if(classical, val)
        return self
