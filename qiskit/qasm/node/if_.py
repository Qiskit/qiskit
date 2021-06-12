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

"""Node for an OPENQASM if statement."""
import warnings

from .node import Node


class If(Node):
    """Node for an OPENQASM if statement.

    children[0] is an id node.
    children[1] is an integer node.
    children[2] is quantum operation node, including U, CX, custom_unitary,
    measure, reset, (and BUG: barrier, if).
    """

    def __init__(self, children):
        """Create the if node."""
        super().__init__("if", children, None)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn(
                "Parameter 'If.qasm(..., prec)' is no longer used and is being " "deprecated.",
                DeprecationWarning,
                2,
            )
        return (
            "if("
            + self.children[0].qasm()
            + "=="
            + str(self.children[1].value)
            + ") "
            + self.children[2].qasm()
        )
