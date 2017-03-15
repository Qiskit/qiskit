"""
Node for an OPENQASM custom gate body.

Author: Jim Challenger
"""
from ._node import Node


class GateBody(Node):
    """Node for an OPENQASM custom gate body.

    children is a list of gate operation nodes.
    These are one of barrier, custom_unitary, U, or CX.
    """

    def __init__(self, children):
        """Create the gatebody node."""
        Node.__init__(self, 'gate_body', children, None)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        s = ""
        for c in self.children:
            s += "  " + c.qasm() + "\n"
        return s

    def calls(self):
        """Return a list of custom gate names in this gate body."""
        lst = []
        for c in self.children:
            if c.type == "custom_unitary":
                lst.append(c.name)
        return lst
