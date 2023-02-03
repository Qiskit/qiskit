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

"""Node for an OPENQASM unary operator."""

import operator

from .node import Node
from .nodeexception import NodeException


VALID_OPERATORS = {
    "+": operator.pos,
    "-": operator.neg,
}


class UnaryOperator(Node):
    """Node for an OPENQASM unary operator.

    This node has no children. The data is in the value field.
    """

    def __init__(self, operation):
        """Create the operator node."""
        super().__init__("unary_operator", None, None)
        self.value = operation

    def operation(self):
        """
        Return the operator as a function f(left, right).
        """
        try:
            return VALID_OPERATORS[self.value]
        except KeyError as ex:
            raise NodeException(f"internal error: undefined prefix '{self.value}'") from ex

    def qasm(self):
        """Return QASM representation."""
        return self.value
