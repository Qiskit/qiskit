# -*- coding: utf-8 -*-

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
Classical register reference object.
"""
import itertools

from qiskit.exceptions import QiskitError
from .register import Register
from .bit import Bit


class ClassicalRegister(Register):
    """Implement a classical register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'c'

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.size)

    def __getitem__(self, key):
        tuple = super().__getitem__(key)
        if isinstance(tuple, list):
            return [ ClBit.from_tuple(bit_tuple) for bit_tuple in tuple ]
        else:
            return  ClBit.from_tuple(tuple)

class ClBit(Bit):
    def __init__(self, register, index):
        if isinstance(register, ClassicalRegister):
            super().__init__(register, index)
        else:
            raise QiskitError('')