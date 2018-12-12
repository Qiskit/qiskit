# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM real number.
"""
from sympy import latex, pi
from sympy.printing.ccode import ccode
from ._node import Node


class Real(Node):
    """Node for an OPENQASM real number.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        """Create the real node."""
        # pylint: disable=redefined-builtin
        Node.__init__(self, "real", None, None)
        self.value = id

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'real', self.value)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        if self.value == pi:
            return "pi"

        return ccode(self.value, precision=prec)

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        # pylint: disable=unused-argument
        return latex(self.value)

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        # pylint: disable=unused-argument
        return self.value

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        # pylint: disable=unused-argument
        return float(self.value.evalf())
