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

"""Node for an OPENQASM external function."""

import sympy
from .node import Node
from .nodeexception import NodeException


class External(Node):
    """Node for an OPENQASM external function.

    children[0] is an id node with the name of the function.
    children[1] is an expression node.
    """

    def __init__(self, children):
        """Create the external node."""
        super().__init__('external', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return self.children[0].qasm(prec) + "(" + \
            self.children[1].qasm(prec) + ")"

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        del prec  # TODO prec ignored
        return sympy.latex(self.sym(nested_scope))

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        op = self.children[0].name
        expr = self.children[1]
        dispatch = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'asin': sympy.asin,
            'acos': sympy.acos,
            'atan': sympy.atan,
            'exp': sympy.exp,
            'ln': sympy.log,
            'sqrt': sympy.sqrt
        }
        if op in dispatch:
            arg = expr.real(nested_scope)
            return dispatch[op](arg)
        else:
            raise NodeException("internal error: undefined external")

    def sym(self, nested_scope=None):
        """Return the corresponding symbolic expression."""
        op = self.children[0].name
        expr = self.children[1]
        dispatch = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'asin': sympy.asin,
            'acos': sympy.acos,
            'atan': sympy.atan,
            'exp': sympy.exp,
            'ln': sympy.log,
            'sqrt': sympy.sqrt
        }
        if op in dispatch:
            arg = expr.sym(nested_scope)
            return dispatch[op](arg)
        else:
            raise NodeException("internal error: undefined external")
