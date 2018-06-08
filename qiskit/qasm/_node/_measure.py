# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM measure statement.
"""
from ._node import Node


class Measure(Node):
    """Node for an OPENQASM measure statement.

    children[0] is a primary node (id or indexedid)
    children[1] is a primary node (id or indexedid)
    """

    def __init__(self, children):
        """Create the measure node."""
        Node.__init__(self, 'measure', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "measure " + self.children[0].qasm(prec) + " -> " + \
               self.children[1].qasm(prec) + ";"
