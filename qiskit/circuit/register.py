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

# pylint: disable=not-callable

"""
Base register reference object.
"""
import re
import itertools

from qiskit.circuit.exceptions import CircuitError


class Register:
    """Implement a generic register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'reg'
    bit_type = None

    def __init__(self, size, name=None):
        """Create a new generic register.
        """

        # validate (or cast) size
        try:
            size = int(size)
        except Exception:
            raise CircuitError("Register size must be castable to an int (%s '%s' was provided)"
                               % (type(size).__name__, size))
        if size <= 0:
            raise CircuitError("Register size must be positive (%s '%s' was provided)"
                               % (type(size).__name__, size))

        # validate (or cast) name
        if name is None:
            name = '%s%i' % (self.prefix, next(self.instances_counter))
        else:
            try:
                name = str(name)
            except Exception:
                raise CircuitError("The circuit name should be castable to a string "
                                   "(or None for autogenerate a name).")
            name_format = re.compile('[a-z][a-zA-Z0-9_]*')
            if name_format.match(name) is None:
                raise CircuitError("%s is an invalid OPENQASM register name." % name)

        self.name = name
        self.size = size

        self._bits = [self.bit_type(self, idx) for idx in range(size)]

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%d, '%s')" % (self.__class__.__qualname__, self.size, self.name)

    def __len__(self):
        """Return register size."""
        return self.size

    def __getitem__(self, key):
        """
        Arg:
            bit_type (Qubit or Clbit): a constructor type return element/s.
            key (int or slice or list): index of the clbit to be retrieved.

        Returns:
            Qubit or Clbit or list(Qubit) or list(Clbit): a Qubit or Clbit instance if
            key is int. If key is a slice, returns a list of these instances.

        Raises:
            CircuitError: if the `key` is not an integer.
            QiskitIndexError: if the `key` is not in the range `(0, self.size)`.
        """
        if not isinstance(key, (int, slice, list)):
            raise CircuitError("expected integer or slice index into register")
        if isinstance(key, slice):
            return self._bits[key]
        elif isinstance(key, list):  # list of qubit indices
            if max(key) < len(self):
                return [self._bits[idx] for idx in key]
            else:
                raise CircuitError('register index out of range')
        else:
            return self._bits[key]

    def __iter__(self):
        for bit in range(self.size):
            yield self[bit]

    def __eq__(self, other):
        """Two Registers are the same if they are of the same type
        (i.e. quantum/classical), and have the same name and size.

        Args:
            other (Register): other Register

        Returns:
            bool: `self` and `other` are equal.
        """
        res = False
        if type(self) is type(other) and \
                self.name == other.name and \
                self.size == other.size:
            res = True
        return res

    def __hash__(self):
        """Make object hashable, based on the name and size to hash."""
        return hash((type(self), self.name, self.size))
