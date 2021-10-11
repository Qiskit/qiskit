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

"""Node for an OPENQASM idlist."""
import warnings

from .node import Node


class IdList(Node):
    """Node for an OPENQASM idlist.

    children is a list of id nodes.
    """

    def __init__(self, children):
        """Create the idlist node."""
        super().__init__('id_list', children, None)

    def size(self):
        """Return the length of the list."""
        return len(self.children)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn('Parameter \'IdList.qasm(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        return ",".join([self.children[j].qasm()
                         for j in range(self.size())])
