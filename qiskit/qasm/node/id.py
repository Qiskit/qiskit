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

"""Node for an OPENQASM id."""

import warnings

from .node import Node
from .nodeexception import NodeException


class Id(Node):
    """Node for an OPENQASM id.

    The node has no children but has fields name, line, and file.
    There is a flag is_bit that is set when XXXXX to help with scoping.
    """

    def __init__(self, id, line, file):
        """Create the id node."""
        # pylint: disable=redefined-builtin
        super().__init__("id", None, None)
        self.name = id
        self.line = line
        self.file = file
        # To help with scoping rules, so we know the id is a bit,
        # this flag is set to True when the id appears in a gate declaration
        self.is_bit = False

    def to_string(self, indent):
        """Print the node with indent."""
        ind = indent * ' '
        print(ind, 'id', self.name)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn('Parameter \'Id.qasm(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        return self.name

    def latex(self, prec=None, nested_scope=None):
        """Return the correspond math mode latex string."""
        if prec is not None:
            warnings.warn('Parameter \'Id.latex(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        if not nested_scope:
            return "\textrm{" + self.name + "}"
        else:
            if self.name not in nested_scope[-1]:
                raise NodeException("Expected local parameter name: ",
                                    "name=%s, " % self.name,
                                    "line=%s, " % self.line,
                                    "file=%s" % self.file)

            return nested_scope[-1][self.name].latex(nested_scope[0:-1])

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        if not nested_scope or self.name not in nested_scope[-1]:
            raise NodeException("Expected local parameter name: ",
                                "name=%s, line=%s, file=%s" % (
                                    self.name, self.line, self.file))
        return nested_scope[-1][self.name].sym(nested_scope[0:-1])

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        if not nested_scope or self.name not in nested_scope[-1]:
            raise NodeException("Expected local parameter name: ",
                                "name=%s, line=%s, file=%s" % (
                                    self.name, self.line, self.file))

        return nested_scope[-1][self.name].real(nested_scope[0:-1])
