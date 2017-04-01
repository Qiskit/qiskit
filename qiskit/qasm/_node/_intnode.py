"""
Node for an OPENQASM integer.

Author: Jim Challenger
"""
from ._node import Node


class Int(Node):
    """Node for an OPENQASM integer.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        """Create the integer node."""
        Node.__init__(self, "int", None, None)
        self.value = id

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'int', self.value)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "%d" % self.value
