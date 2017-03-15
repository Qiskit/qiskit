"""
Node for an OPENQASM indexed id.

Author: Jim Challenger
"""
from ._node import Node


class IndexedId(Node):
    """Node for an OPENQASM indexed id.

    children[0] is an id node.
    children[1] is an integer (not a node).
    """

    def __init__(self, children):
        """Create the indexed id node."""
        Node.__init__(self, 'indexed_id', children, None)
        self.id = children[0]
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = children[1]

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'indexed_id', self.name, self.index)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return self.name + "[%d]" % self.index
