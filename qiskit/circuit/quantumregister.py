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
Quantum register reference object.
"""
import itertools

from qiskit.exceptions import QiskitError
from .register import Register
from .bit import Bit

class QuantumRegister(Register):
    """Implement a quantum register."""
    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'q'

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)

    def __getitem__(self, key):
        """
        Arg:
            key (int|slice|list): index of the qubit to be retrieved.

        Returns:
            tuple[Register, int]: a tuple in the form `(self, key)` if key is int.
                If key is a slice, return a `list((self,key))`.

        Raises:
            QiskitError: if the `key` is not an integer.
            QiskitIndexError: if the `key` is not in the range
                `(0, self.size)`.
        """
        if not isinstance(key, (int, slice, list)):
            raise QiskitError("expected integer or slice index into register")
        if isinstance(key, int) and key < 0:
            key = self.size + key
        self.check_range(key)
        if isinstance(key, slice):
            return [QuBit.from_tuple((self, ind)) for ind in range(*key.indices(len(self)))]
        elif isinstance(key, list):  # list of qubit indices
            if max(key) < len(self):
                return [QuBit.from_tuple((self, ind)) for ind in key]
            else:
                raise QiskitError('register index out of range')
        else:
            return QuBit.from_tuple((self, key))

    def __iter__(self):
        """
        Yields:
            Qubit: an iterator over the qubits in the register.
        """
        for bit in range(self.size):
            yield self[bit]

class QuBit(Bit):
    def __init__(self, register, index):
        if isinstance(register, QuantumRegister):
            super().__init__(register, index)
        else:
            raise QiskitError('')