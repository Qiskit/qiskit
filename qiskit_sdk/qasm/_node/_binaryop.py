"""
Node for an OPENQASM binary operation expression.

Author: Jim Challenger
"""
from ._node import Node


class BinaryOp(Node):
    """Node for an OPENQASM binary operation exprssion.

    children[0] is the operation, as a character.
    children[1] is the left expression.
    children[2] is the right expression.
    """

    def __init__(self, children):
        """Create the binaryop node."""
        Node.__init__(self, 'binop', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "(" + self.children[1].qasm() + self.children[0] + \
               self.children[2].qasm() + ")"
