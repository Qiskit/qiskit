"""
Node for an OPENQASM id.

Author: Jim Challenger
"""
from ._node import Node


class Id(Node):
    """Node for an OPENQASM id.

    The node has no children but has fields name, line, and file.
    There is a flag is_bit that is set when XXXXX to help with scoping.
    """

    def __init__(self, id, line, file):
        """Create the id node."""
        Node.__init__(self, "id", None, None)
        self.name = id
        self.line = line
        self.file = file
        # To help with scoping rules, so we know the id is a bit,
        # this flag is set to True when the id appears in a gate declaration
        self.is_bit = False

    def to_string(self, indent):
        """Print the node with indent."""
        ind = indent * ' '
        print(ind, 'id', self.name)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return self.name
