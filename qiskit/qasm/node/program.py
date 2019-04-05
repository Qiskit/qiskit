# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Node for an OPENQASM program."""

from .node import Node


class Program(Node):
    """Node for an OPENQASM program.

    children is a list of nodes (statements).
    """

    def __init__(self, children):
        """Create the program node."""
        super().__init__('program', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = ""
        for children in self.children:
            string += children.qasm(prec) + "\n"
        return string
