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
Node for an OPENQASM id.
"""
from ._node import Node
from ._nodeexception import NodeException


class Id(Node):
    """Node for an OPENQASM id.

    The node has no children but has fields name, line, and file.
    There is a flag is_bit that is set when XXXXX to help with scoping.
    """

    def __init__(self, id, line, file):
        """Create the id node."""
        # pylint: disable=redefined-builtin
        Node.__init__(self, "id", None, None)
        self.name = id
        self.line = line
        self.file = file
        # To help with scoping rules, so we know the id is a bit,
        # this flag is set to True when the id appears in a gate declaration
        self.is_bit = False

    def to_string(self, indent):
        """Print the node with indent."""
        ind = indent * ' '
        print(ind, 'id', self.name)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        # pylint: disable=unused-argument
        return self.name

    def latex(self, prec=15, nested_scope=None):
        """Return the correspond math mode latex string."""
        if not nested_scope:
            return "\textrm{" + self.name + "}"
        else:
            if self.name not in nested_scope[-1]:
                raise NodeException("Expected local parameter name: ",
                                    "name=%s, " % self.name,
                                    "line=%s, " % self.line,
                                    "file=%s" % self.file)
            else:
                return nested_scope[-1][self.name].latex(prec,
                                                         nested_scope[0:-1])

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        if not nested_scope or self.name not in nested_scope[-1]:
            raise NodeException("Expected local parameter name: ",
                                "name=%s, line=%s, file=%s" % (
                                    self.name, self.line, self.file))
        else:
            return nested_scope[-1][self.name].sym(nested_scope[0:-1])

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        if not nested_scope or self.name not in nested_scope[-1]:
            raise NodeException("Expected local parameter name: ",
                                "name=%s, line=%s, file=%s" % (
                                    self.name, self.line, self.file))
        else:
            return nested_scope[-1][self.name].real(nested_scope[0:-1])
