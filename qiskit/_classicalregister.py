# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Classical register reference object.
"""
import itertools

from ._register import Register


class ClassicalRegister(Register):
    """Implement a classical register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'c'

    def __init__(self, size, name=None):
        """Create a new generic register.
        """
        super().__init__(size=size, name=name)
        self.type = 'creg'

    def __eq__(self, other):
        """Return True if the Register are the same.

        Args:
            other (Register): other Register

        Returns:
            bool: are self and other equal.
        """
        res = False
        if self.type == other.type:
            if self.name == other.name:
                if self.size == other.size:
                    res = True
        return res

    def __hash__(self):
        """Make object is hashable, based on the name and size to hash."""
        return hash(str(self.type)+str(self.name)+str(self.size))

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "creg %s[%d];" % (self.name, self.size)
