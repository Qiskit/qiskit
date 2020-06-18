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

"""Node for an OPENQASM custom gate body."""
import warnings

from .node import Node


class GateBody(Node):
    """Node for an OPENQASM custom gate body.

    children is a list of gate operation nodes.
    These are one of barrier, custom_unitary, U, or CX.
    """

    def __init__(self, children):
        """Create the gatebody node."""
        super().__init__('gate_body', children, None)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn('Parameter \'GateBody.qasm(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        string = ""
        for children in self.children:
            string += "  " + children.qasm() + "\n"
        return string

    def calls(self):
        """Return a list of custom gate names in this gate body."""
        lst = []
        for children in self.children:
            if children.type == "custom_unitary":
                lst.append(children.name)
        return lst
