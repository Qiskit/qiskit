"""
Node for an OPENQASM prefix expression.

Author: Jim Challenger
"""
from ._node import Node


class Prefix(Node):
    """Node for an OPENQASM prefix expression.

    children[0] is a prefix string such as '-'.
    children[1] is an expression node.
    """

    def __init__(self, children):
        """Create the prefix node."""
        Node.__init__(self, 'prefix', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return self.children[0] + "(" + self.children[1].qasm() + ")"
