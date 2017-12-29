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
Node for an OPENQASM if statement.
"""
from ._node import Node


class If(Node):
    """Node for an OPENQASM if statement.

    children[0] is an id node.
    children[1] is an integer node.
    children[2] is quantum operation node, including U, CX, custom_unitary,
    measure, reset, (and BUG: barrier, if).
    """

    def __init__(self, children):
        """Create the if node."""
        Node.__init__(self, 'if', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "if(" + self.children[0].qasm(prec) + "==" \
               + str(self.children[1].value) + ") " + \
               self.children[2].qasm(prec)
