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
