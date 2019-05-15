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
Base register reference object.
"""
import re
import logging
import itertools

from qiskit.exceptions import QiskitError, QiskitIndexError

logger = logging.getLogger(__name__)


class Register:
    """Implement a generic register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'reg'

    def __init__(self, size, name=None):
        """Create a new generic register.
        """

        # validate (or cast) size
        try:
            size = int(size)
        except Exception:
            raise QiskitError("Register size must be castable to an int (%s '%s' was provided)"
                              % (type(size).__name__, size))
        if size <= 0:
            raise QiskitError("Register size must be positive (%s '%s' was provided)"
                              % (type(size).__name__, size))

        # validate (or cast) name
        if name is None:
            name = '%s%i' % (self.prefix, next(self.instances_counter))
        else:
            try:
                name = str(name)
            except Exception:
                raise QiskitError("The circuit name should be castable to a string "
                                  "(or None for autogenerate a name).")
            name_format = re.compile('[a-z][a-zA-Z0-9_]*')
            if name_format.match(name) is None:
                raise QiskitError("%s is an invalid OPENQASM register name." % name)

        self.name = name
        self.size = size

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%d, '%s')" % (self.__class__.__qualname__,
                                 self.size, self.name)

    def __len__(self):
        """Return register size"""
        return self.size

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if isinstance(j, int):
            if j < 0 or j >= self.size:
                raise QiskitIndexError("register index out of range")
            if isinstance(j, slice):
                if j.start < 0 or j.stop >= self.size or (j.step is not None and
                                                          j.step <= 0):
                    raise QiskitIndexError("register index slice out of range")

    def __getitem__(self, key):
        """
        Arg:
            key (int|slice|list): index of the bit/qubit to be retrieved.

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
            return [(self, ind) for ind in range(*key.indices(len(self)))]
        elif isinstance(key, list):  # list of qubit indices
            if max(key) < len(self):
                return [(self, ind) for ind in key]
            else:
                raise QiskitError('register index out of range')
        else:
            return self, key

    def __iter__(self):
        """
        Returns:
            iterator: an iterator over the bits/qubits of the register, in the
                form `tuple (Register, int)`.
        """
        return zip([self] * self.size, range(self.size))

    def __eq__(self, other):
        """Two Registers are the same if they are of the same type
        (i.e. quantum/classical), and have the same name and size.

        Args:
            other (Register): other Register

        Returns:
            bool: are self and other equal.
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
