# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM primarylist.
"""
from ._node import Node


class PrimaryList(Node):
    """Node for an OPENQASM primarylist.

    children is a list of primary nodes. Primary nodes are indexedid or id.
    """

    def __init__(self, children):
        """Create the primarylist node."""
        Node.__init__(self, 'primary_list', children, None)

    def size(self):
        """Return the size of the list."""
        return len(self.children)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return ",".join([self.children[j].qasm(prec)
                         for j in range(self.size())])
