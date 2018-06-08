# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM U statement.
"""
from ._node import Node


class UniversalUnitary(Node):
    """Node for an OPENQASM U statement.

    children[0] is an expressionlist node.
    children[1] is a primary node (id or indexedid).
    """

    def __init__(self, children):
        """Create the U node."""
        Node.__init__(self, 'universal_unitary', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "U(" + self.children[0].qasm(prec) + ") " + \
               self.children[1].qasm(prec) + ";"
