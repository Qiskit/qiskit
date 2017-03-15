"""
Node for an OPENQASM file identifier/version statement.

Author: Jim Challenger
"""
from ._node import Node


class Magic(Node):
    """Node for an OPENQASM file identifier/version statement ("magic number").

    children[0] is a floating point number (not a node).
    """

    def __init__(self, children):
        """Create the version node."""
        Node.__init__(self, 'magic', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "OPENQASM %.1f;" % self.children[0]
