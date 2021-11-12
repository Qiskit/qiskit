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

"""Node for an OPENQASM primarylist."""
from .node import Node


class PrimaryList(Node):
    """Node for an OPENQASM primarylist.

    children is a list of primary nodes. Primary nodes are indexedid or id.
    """

    def __init__(self, children):
        """Create the primarylist node."""
        super().__init__("primary_list", children, None)

    def size(self):
        """Return the size of the list."""
        return len(self.children)

    def qasm(self):
        """Return the corresponding OPENQASM string."""
        return ",".join([self.children[j].qasm() for j in range(self.size())])
