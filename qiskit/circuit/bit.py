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
Quantum bit and Classical bit objects.
"""
from qiskit.circuit.exceptions import CircuitError


class Bit:
    """Implement a generic bit."""

    def __init__(self, register, index):
        """Create a new generic bit.
        """
        try:
            index = int(index)
        except Exception:
            raise CircuitError("index needs to be castable to an int: type %s was provided" %
                               type(index))

        if index < 0:
            index += register.size

        if index >= register.size:
            raise CircuitError("index must be under the size of the register: %s was provided" %
                               index)

        self.register = register
        self.index = index

    def __repr__(self):
        """Return the official string representing the bit."""
        return "%s(%s, %s)" % (self.__class__.__name__, self.register, self.index)

    def __hash__(self):
        return hash((self.register, self.index))

    def __eq__(self, other):
        if isinstance(other, Bit):
            return other.index == self.index and other.register == self.register
        return False
