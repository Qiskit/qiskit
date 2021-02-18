# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base node object for the OPENQASM syntax tree."""


class Node:
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
        ind = indent * " "
        if self.root:
            print(ind, self.type, "---", self.root)
        else:
            print(ind, self.type)
        indent = indent + 3
        ind = indent * " "
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
