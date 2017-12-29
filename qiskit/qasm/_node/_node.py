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
Base node object for the OPENQASM syntax tree.
"""


class Node(object):
    """Base node object for the OPENQASM syntax tree."""

    def __init__(self, type, children=None, root=None):
        """Construct a new node object."""
        # pylint: disable=redefined-builtin
        self.type = type
        if children:
            self.children = children
        else:
            self.children = []
        self.root = root
        # True if this node is an expression node, False otherwise
        self.expression = False

    def is_expression(self):
        """Return True if this is an expression node."""
        return self.expression

    def add_child(self, node):
        """Add a child node."""
        self.children.append(node)

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        if self.root:
            print(ind, self.type, '---', self.root)
        else:
            print(ind, self.type)
        indent = indent + 3
        ind = indent * ' '
        for children in self.children:
            if children is None:
                print("OOPS! type of parent is", type(self))
                print(self.children)
            if isinstance(children, str):
                print(ind, children)
            elif isinstance(children, int):
                print(ind, str(children))
            elif isinstance(children, float):
                print(ind, str(children))
            else:
                children.to_string(indent)
