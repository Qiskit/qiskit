"""
Node for an OPENQASM real number.

Author: Jim Challenger
"""
from ._node import Node


class Real(Node):
    """Node for an OPENQASM real number.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        """Create the real node."""
        Node.__init__(self, "real", None, None)
        self.value = id

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'real', self.value)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "%0.15f" % self.value  # TODO: control the precision
