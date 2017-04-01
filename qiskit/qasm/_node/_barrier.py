"""
Node for an OPENQASM barrier statement.

Author: Jim Challenger
"""
from ._node import Node


class Barrier(Node):
    """Node for an OPENQASM barrier statement.

    children[0] is a primarylist node.
    """

    def __init__(self, children):
        """Create the barrier node."""
        Node.__init__(self, 'barrier', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "barrier " + self.children[0].qasm() + ";"
