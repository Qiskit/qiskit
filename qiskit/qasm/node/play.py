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

"""Node for a pulse play definition."""

from .node import Node


class Play(Node):
    """Node for an OPENQASM gate definition.

    children[0] is an id node.
    children[1] is a channel node.
    """

    def __init__(self, children):
        """Create the gate node."""
        super().__init__('play', children, None)
        self.arguments = children[0]  # pylint: disable=invalid-name
        self.channel = children[1]

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        string = "play " + self.arguments.qasm(prec) + ' ' + self.channel.qasm(prec) + ';'
        return string
