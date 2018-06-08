# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Node for an OPENQASM custom gate statement.
"""
from ._node import Node


class CustomUnitary(Node):
    """Node for an OPENQASM custom gate statement.

    children[0] is an id node.
    children[1] is an exp_list (if len==3) or primary_list.
    children[2], if present, is a primary_list.

    Has properties:
    .id = id node
    .name = gate name string
    .arguments = None or exp_list node
    .bitlist = primary_list node
    """

    def __init__(self, children):
        """Create the custom gate node."""
        Node.__init__(self, 'custom_unitary', children, None)
        self.id = children[0]
        self.name = self.id.name
        if len(children) == 3:
            self.arguments = children[1]
            self.bitlist = children[2]
        else:
            self.arguments = None
            self.bitlist = children[1]

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = self.name
        if self.arguments is not None:
            string += "(" + self.arguments.qasm(prec) + ")"
        string += " " + self.bitlist.qasm(prec) + ";"
        return string
