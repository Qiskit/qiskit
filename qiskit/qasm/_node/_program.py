# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM program.
"""
from ._node import Node


class Program(Node):
    """Node for an OPENQASM program.

    children is a list of nodes (statements).
    """

    def __init__(self, children):
        """Create the program node."""
        Node.__init__(self, 'program', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = ""
        previous_stmt_line = 1
        for children in self.children:
            if children.line != previous_stmt_line:
                 string += "\n"
                 previous_stmt_line = children.line
            string += children.qasm(prec)
        return string
