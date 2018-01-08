# -*- coding: utf-8 -*-

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
