# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM custom gate body.
"""
from ._node import Node


class GateBody(Node):
    """Node for an OPENQASM custom gate body.

    children is a list of gate operation nodes.
    These are one of barrier, custom_unitary, U, or CX.
    """

    def __init__(self, children):
        """Create the gatebody node."""
        Node.__init__(self, 'gate_body', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = ""
        for children in self.children:
            string += "  " + children.qasm(prec) + "\n"
        return string

    def calls(self):
        """Return a list of custom gate names in this gate body."""
        lst = []
        for children in self.children:
            if children.type == "custom_unitary":
                lst.append(children.name)
        return lst
