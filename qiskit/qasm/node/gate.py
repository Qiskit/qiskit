# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Node for an OPENQASM gate definition."""
from .node import Node


class Gate(Node):
    """Node for an OPENQASM gate definition.

    children[0] is an id node.
    If len(children) is 3, children[1] is an idlist node,
    and children[2] is a gatebody node.
    Otherwise, children[1] is an expressionlist node,
    children[2] is an idlist node, and children[3] is a gatebody node.
    """

    def __init__(self, children):
        """Create the gate node."""
        super().__init__("gate", children, None)
        self.id = children[0]  # pylint: disable=invalid-name
        # The next three fields are required by the symbtab
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file

        if len(children) == 3:
            self.arguments = None
            self.bitlist = children[1]
            self.body = children[2]
        else:
            self.arguments = children[1]
            self.bitlist = children[2]
            self.body = children[3]

    def n_args(self):
        """Return the number of parameter expressions."""
        if self.arguments:
            return self.arguments.size()
        return 0

    def n_bits(self):
        """Return the number of qubit arguments."""
        return self.bitlist.size()

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        string = "gate " + self.name
        if self.arguments is not None:
            string += "(" + self.arguments.qasm() + ")"
        string += " " + self.bitlist.qasm() + "\n"
        string += "{\n" + self.body.qasm() + "}"
        return string
