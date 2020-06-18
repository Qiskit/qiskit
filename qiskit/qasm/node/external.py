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

import warnings
import numpy as np

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

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn('Parameter \'External.qasm(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        return self.children[0].qasm() + "(" + self.children[1].qasm() + ")"

    def latex(self, prec=None, nested_scope=None):
        """Return the corresponding math mode latex string."""
        if prec is not None:
            warnings.warn('Parameter \'External.latex(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        if nested_scope is not None:
            warnings.warn('Parameter \'External.latex(..., nested_scope)\' is no longer used and '
                          'is being deprecated.', DeprecationWarning, 2)
        try:
            from pylatexenc.latexencode import utf8tolatex
        except ImportError:
            raise ImportError("To export latex from qasm "
                              "pylatexenc needs to be installed. Run "
                              "'pip install pylatexenc' before using this "
                              "method.")
        return utf8tolatex(self.sym())

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        op = self.children[0].name
        expr = self.children[1]
        dispatch = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'asin': np.arcsin,
            'acos': np.arccos,
            'atan': np.arctan,
            'exp': np.exp,
            'ln': np.log,
            'sqrt': np.sqrt
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
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'asin': np.arcsin,
            'acos': np.arccos,
            'atan': np.arctan,
            'exp': np.exp,
            'ln': np.log,
            'sqrt': np.sqrt
        }
        if op in dispatch:
            arg = expr.sym(nested_scope)
            return dispatch[op](arg)
        else:
            raise NodeException("internal error: undefined external")
