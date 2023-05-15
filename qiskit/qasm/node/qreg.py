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

"""Node for an OPENQASM qreg statement."""
from .node import Node


class Qreg(Node):
    """Node for an OPENQASM qreg statement.

    children[0] is an indexedid node.
    """

    def __init__(self, children):
        """Create the qreg node."""
        super().__init__("qreg", children, None)
        # This is the indexed id, the full "id[n]" object
        self.id = children[0]  # pylint: disable=invalid-name
        # Name of the qreg
        self.name = self.id.name
        # Source line number
        self.line = self.id.line
        # Source file name
        self.file = self.id.file
        # Size of the register
        self.index = self.id.index

    def to_string(self, indent):
        """Print the node data, with indent."""
        ind = indent * " "
        print(ind, "qreg")
        self.children[0].to_string(indent + 3)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return "qreg " + self.id.qasm() + ";"
