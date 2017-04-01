"""
Base node object for the OPENQASM syntax tree.

Author: Jim Challenger
"""


class Node(object):
    """Base node object for the OPENQASM syntax tree."""

    def __init__(self, type, children=None, root=None):
        """Construct a new node object."""
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

    def add_child(self, n):
        """Add a child node."""
        self.children.append(n)

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        if self.root:
            print(ind, self.type, '---', self.root)
        else:
            print(ind, self.type)
        indent = indent + 3
        ind = indent * ' '
        for c in self.children:
            if c is None:
                print("OOPS! type of parent is", type(self))
                print(self.children)
            if type(c) is str:
                print(ind, c)
            elif type(c) is int:
                print(ind, str(c))
            elif type(c) is float:
                print(ind, str(c))
            else:
                c.to_string(indent)
