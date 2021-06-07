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

"""Node for an OPENQASM indexed id."""

import warnings

from .node import Node


class IndexedId(Node):
    """Node for an OPENQASM indexed id.

    children[0] is an id node.
    children[1] is an Int node.
    """

    def __init__(self, children):
        """Create the indexed id node."""
        super().__init__("indexed_id", children, None)
        self.id = children[0]  # pylint: disable=invalid-name
        self.name = self.id.name
        self.line = self.id.line
        self.file = self.id.file
        self.index = children[1].value

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * " "
        print(ind, "indexed_id", self.name, self.index)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn(
                "Parameter 'IndexedId.qasm(..., prec)' is no longer used and is being "
                "deprecated.",
                DeprecationWarning,
                2,
            )
        return self.name + "[%d]" % self.index
