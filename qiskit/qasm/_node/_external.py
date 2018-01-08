# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Node for an OPENQASM external function.
"""
import sympy
from ._node import Node
from ._nodeexception import NodeException


class External(Node):
    """Node for an OPENQASM external function.

    children[0] is an id node with the name of the function.
    children[1] is an expression node.
    """

    def __init__(self, children):
        """Create the external node."""
        Node.__init__(self, 'external', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return self.children[0].qasm(prec) + "(" + \
            self.children[1].qasm(prec) + ")"

    def latex(self, prec=15, nested_scope=None):
        """Return the corresponding math mode latex string."""
        # pylint: disable=unused-argument
        # TODO prec ignored
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
