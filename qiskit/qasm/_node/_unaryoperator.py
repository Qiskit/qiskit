# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM unary operator.
"""
import operator

from ._node import Node
from ._nodeexception import NodeException


VALID_OPERATORS = {
    '+': operator.pos,
    '-': operator.neg,
}


class UnaryOperator(Node):
    """Node for an OPENQASM unary operator.

    This node has no children. The data is in the value field.
    """
    def __init__(self, operation):
        """Create the operator node."""
        Node.__init__(self, 'unary_operator', None, None)
        self.value = operation

    def operation(self):
        """
        Return the operator as a function f(left, right).
        """
        try:
            return VALID_OPERATORS[self.value]
        except KeyError:
            raise NodeException("internal error: undefined prefix '%s'" %
                                self.value)

    def qasm(self, prec=15):
        """Return QASM representation."""
        # pylint: disable=unused-argument
        return self.value
