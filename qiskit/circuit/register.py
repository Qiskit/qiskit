# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Base register reference object.
"""
import re
import logging
import itertools

from qiskit.qiskiterror import QiskitError, QiskitIndexError

logger = logging.getLogger(__name__)


class Register(object):
    """Implement a generic register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'reg'

    def __init__(self, size, name=None):
        """Create a new generic register.
        """

        if name is None:
            name = '%s%i' % (self.prefix, next(self.instances_counter))

        if not isinstance(name, str):
            raise QiskitError("The circuit name should be a string "
                              "(or None for autogenerate a name).")

        test = re.compile('[a-z][a-zA-Z0-9_]*')
        if test.match(name) is None:
            raise QiskitError("%s is an invalid OPENQASM register name." % name)

        self.name = name
        self.size = size
        if size <= 0:
            raise QiskitError("register size must be positive")

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%d, '%s')" % (self.__class__.__qualname__,
                                 self.size, self.name)

    def __len__(self):
        """Return register size"""
        return self.size

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.size:
            raise QiskitIndexError("register index out of range")

    def __getitem__(self, key):
        """
        Arg:
            key (int): index of the bit/qubit to be retrieved.

        Returns:
            tuple[Register, int]: a tuple in the form `(self, key)`.

        Raises:
            QiskitError: if the `key` is not an integer.
            QiskitIndexError: if the `key` is not in the range
                `(0, self.size)`.
        """
        if not isinstance(key, int):
            raise QiskitError("expected integer index into register")
        self.check_range(key)
        return self, key

    def __iter__(self):
        """
        Returns:
            iterator: an iterator over the bits/qubits of the register, in the
                form `tuple (Register, int)`.
        """
        return zip([self]*self.size, range(self.size))

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
        return hash(str(type(self)) + self.name + str(self.size))
