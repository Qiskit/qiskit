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
Node for an OPENQASM U statement.
"""
from ._node import Node


class UniversalUnitary(Node):
    """Node for an OPENQASM U statement.

    children[0] is an expressionlist node.
    children[1] is a primary node (id or indexedid).
    """

    def __init__(self, children):
        """Create the U node."""
        Node.__init__(self, 'universal_unitary', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "U(" + self.children[0].qasm(prec) + ") " + \
               self.children[1].qasm(prec) + ";"
