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

"""Node for an OPENQASM expression list."""
from .node import Node


class ExpressionList(Node):
    """Node for an OPENQASM expression list.

    children are expression nodes.
    """

    def __init__(self, children):
        """Create the expression list node."""
        super().__init__("expression_list", children, None)

    def size(self):
        """Return the number of expressions."""
        return len(self.children)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return ",".join([self.children[j].qasm() for j in range(self.size())])
