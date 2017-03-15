"""
Node for an OPENQASM external function.

Author: Jim Challenger
"""
from ._node import Node


class External(Node):
    """Node for an OPENQASM external function.

    children[0] is an id node with the name of the function.
    children[1] is an expression node.
    """

    def __init__(self, children):
        """Create the external node."""
        Node.__init__(self, 'external', children, None)
