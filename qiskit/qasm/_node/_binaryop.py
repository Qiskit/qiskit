# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM binary operation expression.
"""
import sympy

from ._node import Node


class BinaryOp(Node):
    """Node for an OPENQASM binary operation expression.

    children[0] is the operation, as a binary operator node.
    children[1] is the left expression.
    children[2] is the right expression.
    """

    def __init__(self, children):
        """Create the binaryop node."""
        Node.__init__(self, 'binop', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "(" + self.children[1].qasm(prec) + self.children[0].value + \
               self.children[2].qasm(prec) + ")"

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        # pylint: disable=unused-argument
        # TODO prec ignored
        return sympy.latex(self.sym(nested_scope))

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        operation = self.children[0].operation()
        lhs = self.children[1].real(nested_scope)
        rhs = self.children[2].real(nested_scope)
        return operation(lhs, rhs)

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        operation = self.children[0].operation()
        lhs = self.children[1].sym(nested_scope)
        rhs = self.children[2].sym(nested_scope)
        return operation(lhs, rhs)
