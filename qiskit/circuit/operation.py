# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Operation object
"""

from collections.abc import Sequence


class Operation(Sequence):
    """Class representation of a circuit operation

    An operation is an instruction with it's operands.
    """

    __slots__ = ("instruction", "qargs", "cargs")

    def __init__(self, instruction, qargs=None, cargs=None):
        """Initialize a new instruction object

        Args:
            instruction (qiskit.circuit.Instruction): The instruction object for
                the operation
            qargs (list): A list of :class:`~qiskit.circuit.Qubit` objects that
                the instruction runs on
            cargs (list): A list of :class:`~qiskit.circuit.Clbit` objects that
                the instruction runs on
        """
        self.instruction = instruction
        if qargs is None:
            self.qargs = []
        else:
            self.qargs = qargs
        if cargs is None:
            self.cargs = []
        else:
            self.cargs = cargs

    def __len__(self):
        return 3

    def __getitem__(self, index):
        if index == 0:
            return self.instruction
        if index == 1:
            return self.qargs
        if index == 2:
            return self.cargs
        if isinstance(index, slice):
            out_items = (self.instruction, self.qargs, self.cargs)
            return out_items.__getitem__(index)
        raise IndexError("Index %s is out of range" % index)

    def __eq__(self, other):
        if isinstance(other, tuple):
            if other[0] == self.instruction and other[1] == self.qargs and other[2] == self.cargs:
                return True
            return False
        elif isinstance(other, Operation):
            if (
                self.instruction == other.instruction
                and self.qargs == other.qargs
                and self.cargs == other.cargs
            ):
                return True
            return False
        else:
            return False
