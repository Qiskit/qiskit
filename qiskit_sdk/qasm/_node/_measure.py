"""
Node for an OPENQASM measure statement.

Author: Jim Challenger
"""
from ._node import Node


class Measure(Node):
    """Node for an OPENQASM measure statement.

    children[0] is a primary node (id or indexedid)
    children[1] is a primary node (id or indexedid)
    """

    def __init__(self, children):
        """Create the measure node."""
        Node.__init__(self, 'measure', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "measure " + self.children[0].qasm() + " -> " + \
               self.children[1].qasm() + ";"
