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


class Bit():
    """Implement a generic bit."""

    __slots__ = {'_register', '_index'}

    _instances = dict()

    def __new__(cls, register, index):
        key = (cls, register, index)
        if key in cls._instances:
            return cls._instances[key]

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

        key = (cls, register, index)
        if key in cls._instances:
            return cls._instances[key]

        obj = super().__new__(cls)

        obj._register = register
        obj._index = index

        cls._instances[key] = obj
        return obj

    def __getnewargs__(self):
        return (self._register, self._index)  # pylint: disable=no-member

    @property
    def register(self):
        """Returns the register containing the bit."""
        return self._register  # pylint: disable=no-member

    @property
    def index(self):
        """Returns the index of the bit in its containing register."""
        return self._index  # pylint: disable=no-member

    def __repr__(self):
        """Return the official string representing the bit."""
        return "%s(%s, %s)" % (self.__class__.__name__, self.register, self.index)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self
