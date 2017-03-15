"""
Node for an OPENQASM expression list.

Author: Jim Challenger
"""
from ._node import Node


class ExpressionList(Node):
    """Node for an OPENQASM expression list.

    children are expression nodes.
    """

    def __init__(self, children):
        """Create the expression list node."""
        Node.__init__(self, 'expression_list', children, None)

    def size(self):
        """Return the number of expressions."""
        return len(self.children)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return ",".join([self.children[j].qasm() for j in range(self.size())])
