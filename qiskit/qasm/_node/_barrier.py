# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM barrier statement.
"""
from ._node import Node


class Barrier(Node):
    """Node for an OPENQASM barrier statement.

    children[0] is a primarylist node.
    """

    def __init__(self, children):
        """Create the barrier node."""
        Node.__init__(self, 'barrier', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "barrier " + self.children[0].qasm(prec) + ";"
