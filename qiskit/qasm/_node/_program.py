"""
Node for an OPENQASM program.

Author: Jim Challenger
"""
from ._node import Node


class Program(Node):
    """Node for an OPENQASM program.

    children is a list of nodes (statements).
    """

    def __init__(self, children):
        """Create the program node."""
        Node.__init__(self, 'program', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        string = ""
        for children in self.children:
            string += children.qasm() + "\n"
        return string
