# -*- coding: utf-8 -*-

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

"""Node for an OPENQASM U statement."""

from .node import Node


class UniversalUnitary(Node):
    """Node for an OPENQASM U statement.

    children[0] is an expressionlist node.
    children[1] is a primary node (id or indexedid).
    """

    def __init__(self, children):
        """Create the U node."""
        super().__init__('universal_unitary', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "U(" + self.children[0].qasm(prec) + ") " + \
               self.children[1].qasm(prec) + ";"
