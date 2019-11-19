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

"""Node for a pulse acquire statement."""

from .node import Node


class Acquire(Node):
    """Node for a pulse acquire statement.

    children[0] is a primary channel indexedid node
    children[1] is a primary creg indexedid node
    """

    def __init__(self, children):
        """Create the acquire node."""
        super().__init__('acquire', children, None)
        self.channel = self.children[0]
        self.creg = self.children[1]

    def qasm(self, prec=15):
        """Return the corresponding pulse string."""
        return "acquire " + self.channel.qasm(prec) + " " + \
               self.creg.qasm(prec) + ";"
