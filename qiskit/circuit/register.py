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

from qiskit.exceptions import QiskitError


class Register(list):
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
        for item in range(size):
            self.append(self.bit_type(self, item))

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%d, '%s')" % (self.__class__.__qualname__, len(self), self.name)

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
                len(self) == len(other):
            res = True
        return res
