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

    __slots__ = {'_register', '_index', '_hash'}

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

        self._register = register
        self._index = index
        self._update_hash()

    def _update_hash(self):
        self._hash = hash((self._register, self._index))

    @property
    def register(self):
        """Get bit's register."""
        return self._register

    @register.setter
    def register(self, value):
        """Set bit's register."""
        self._register = value
        self._update_hash()

    @property
    def index(self):
        """Get bit's index."""
        return self._index

    @index.setter
    def index(self, value):
        """Set bit's index."""
        self._index = value
        self._update_hash()

    def __repr__(self):
        """Return the official string representing the bit."""
        return "%s(%s, %s)" % (self.__class__.__name__, self._register, self._index)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Bit):
            return other._index == self._index and other._register == self._register
        return False
