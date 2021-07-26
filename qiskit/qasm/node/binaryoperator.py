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

"""Node for an OPENQASM binary operator."""

import operator
import warnings

from .node import Node
from .nodeexception import NodeException


VALID_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}


class BinaryOperator(Node):
    """Node for an OPENQASM binary operator.

    This node has no children. The data is in the value field.
    """

    def __init__(self, operation):
        """Create the operator node."""
        super().__init__("operator", None, None)
        self.value = operation

    def operation(self):
        """
        Return the operator as a function f(left, right).
        """
        try:
            return VALID_OPERATORS[self.value]
        except KeyError as ex:
            raise NodeException(f"internal error: undefined operator '{self.value}'") from ex

    def qasm(self, prec=None):
        """Return the QASM representation."""
        if prec is not None:
            warnings.warn(
                "Parameter 'BinaryOperator.qasm(..., prec)' is no longer used and is "
                "being deprecated.",
                DeprecationWarning,
                2,
            )
        return self.value
