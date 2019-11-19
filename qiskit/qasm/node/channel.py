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

"""Node for an OPENQASM qreg statement."""

from .node import Node


class Channel(Node):
    """Node for an OPENQASM qreg statement.

    children[0] is an indexedid node.
    """

    def __init__(self, children):
        """Create the channel node."""
        super().__init__('channel', children, None)

        self.id = children[0]  # pylint: disable=invalid-name
        # Name of the qreg
        self.name = self.id.name
        # Source line number
        self.line = self.id.line
        # Source file name
        self.file = self.id.file
        # Size of the register
        self.index = self.id.index

    def qasm(self, prec=15):
        """Return the corresponding pulse string."""
        return "channel " + self.id.qasm(prec) + ";"
