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

"""Node for an OPENQASM binary operation expression."""

import warnings

from qiskit.exceptions import MissingOptionalLibraryError
from .node import Node


class BinaryOp(Node):
    """Node for an OPENQASM binary operation expression.

    children[0] is the operation, as a binary operator node.
    children[1] is the left expression.
    children[2] is the right expression.
    """

    def __init__(self, children):
        """Create the binaryop node."""
        super().__init__("binop", children, None)

    def qasm(self, prec=None, nested_scope=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn(
                "Parameter 'BinaryOp.qasm(..., prec)' is no longer "
                "used and is being deprecated.",
                DeprecationWarning,
                2,
            )
        if nested_scope is not None:
            warnings.warn(
                "Parameter 'BinaryOp.qasm(..., nested_scope)' is no longer "
                "used and is being deprecated.",
                DeprecationWarning,
                2,
            )
        return (
            "(" + self.children[1].qasm() + self.children[0].value + self.children[2].qasm() + ")"
        )

    def latex(self, prec=None, nested_scope=None):
        """Return the corresponding math mode latex string."""
        if prec is not None:
            warnings.warn(
                "Parameter 'BinaryOp.latex(..., prec)' is no longer used "
                "and is being deprecated.",
                DeprecationWarning,
                2,
            )
        if nested_scope is not None:
            warnings.warn(
                "Parameter 'BinaryOp.latex(..., nested_scope)' is no longer used "
                "and is being deprecated.",
                DeprecationWarning,
                2,
            )
        try:
            from pylatexenc.latexencode import utf8tolatex
        except ImportError as ex:
            raise MissingOptionalLibraryError(
                "pylatexenc", "latex-from-qasm exporter", "pip install pylatexenc"
            ) from ex
        return utf8tolatex(self.sym())

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        if nested_scope is not None:
            warnings.warn(
                "Parameter 'BinaryOp.real(..., nested_scope)' is no longer used and is"
                " being deprecated.",
                DeprecationWarning,
            )
        operation = self.children[0].operation()
        lhs = self.children[1].real()
        rhs = self.children[2].real()
        return operation(lhs, rhs)

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        operation = self.children[0].operation()
        lhs = self.children[1].sym(nested_scope)
        rhs = self.children[2].sym(nested_scope)
        return operation(lhs, rhs)
