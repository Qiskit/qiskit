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

"""Node for a pulse definition."""

from .node import Node


class Pulse(Node):
    """Node for an OPENQASM gate definition.

    children[0] is an id node.
    """

    def __init__(self, children):
        """Create the pulse node.

        children[0] is an id.
        children[1] is an expressionlist of complex.
        """
        super().__init__('pulse', children, None)
        # This is the indexed id, the full "id[n]" object
        self.id = children[0]  # pylint: disable=invalid-name
        # Name of the creg
        self.name = self.id.name
        # Source line number
        self.line = self.id.line
        # Source file name
        self.file = self.id.file
        # Size of the register
        # self.index = self.id.index

        self.samples = children[1]

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "pulse " + self.id.qasm(prec) + " [" + self.samples.qasm(prec) + "];"
