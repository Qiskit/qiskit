# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM prefix expression.
"""
import sympy

from ._node import Node


class Prefix(Node):
    """Node for an OPENQASM prefix expression.

    children[0] is a unary operator node.
    children[1] is an expression node.
    """

    def __init__(self, children):
        """Create the prefix node."""
        Node.__init__(self, 'prefix', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return self.children[0].value + "(" + self.children[1].qasm(prec) + ")"

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        # pylint: disable=unused-argument
        # TODO prec ignored
        return sympy.latex(self.sym(nested_scope))

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        operation = self.children[0].operation()
        expr = self.children[1].real(nested_scope)
        return operation(expr)

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        operation = self.children[0].operation()
        expr = self.children[1].sym(nested_scope)
        return operation(expr)
