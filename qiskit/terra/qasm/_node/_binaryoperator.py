# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM binary operator.
"""
import operator

from ._node import Node
from ._nodeexception import NodeException


VALID_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '^': operator.pow
}


class BinaryOperator(Node):
    """Node for an OPENQASM binary operator.

    This node has no children. The data is in the value field.
    """
    def __init__(self, operation):
        """Create the operator node."""
        Node.__init__(self, 'operator', None, None)
        self.value = operation

    def operation(self):
        """
        Return the operator as a function f(left, right).
        """
        try:
            return VALID_OPERATORS[self.value]
        except KeyError:
            raise NodeException("internal error: undefined operator '%s'" %
                                self.value)

    def qasm(self, prec=15):
        """Return the QASM representation."""
        # pylint: disable=unused-argument
        return self.value
