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
