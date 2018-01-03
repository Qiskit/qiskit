# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

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
Node for an OPENQASM opaque gate declaration.
"""
from ._node import Node


class Opaque(Node):
    """Node for an OPENQASM opaque gate declaration.

    children[0] is an id node.
    If len(children) is 3, children[1] is an expressionlist node,
    and children[2] is an idlist node.
    Otherwise, children[1] is an idlist node.
    """

    def __init__(self, children):
        """Create the opaque gate node."""
        Node.__init__(self, 'opaque', children, None)
        self.id = children[0]
        # The next three fields are required by the symbtab
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        if len(children) == 3:
            self.arguments = children[1]
            self.bitlist = children[2]
        else:
            self.arguments = None
            self.bitlist = children[1]

    def n_args(self):
        """Return the number of parameter expressions."""
        if self.arguments:
            return self.arguments.size()
        return 0

    def n_bits(self):
        """Return the number of qubit arguments."""
        return self.bitlist.size()

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = "opaque %s" % self.name
        if self.arguments is not None:
            string += "(" + self.arguments.qasm(prec) + ")"
        string += " " + self.bitlist.qasm(prec) + ";"
        return string
