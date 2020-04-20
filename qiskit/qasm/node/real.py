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

"""Node for an OPENQASM real number."""

import warnings
import numpy as np

from .node import Node


class Real(Node):
    """Node for an OPENQASM real number.

    This node has no children. The data is in the value field.
    """

    def __init__(self, id):
        """Create the real node."""
        # pylint: disable=redefined-builtin
        super().__init__("real", None, None)
        self.value = id

    def to_string(self, indent):
        """Print with indent."""
        ind = indent * ' '
        print(ind, 'real', self.value)

    def qasm(self, prec=None):
        """Return the corresponding OPENQASM string."""
        if prec is not None:
            warnings.warn('Parameter \'Real.qasm(..., prec)\' is no longer used and'
                          ' is being deprecated.', DeprecationWarning, 2)
        if self.value == np.pi:
            return "pi"

        return str(np.round(float(self.value)))

    def latex(self, prec=None, nested_scope=None):
        """Return the corresponding math mode latex string."""
        if prec is not None:
            warnings.warn('Parameter \'Real.latex(..., prec)\' is no longer used and is being '
                          'deprecated.', DeprecationWarning, 2)
        if nested_scope is not None:
            warnings.warn('Parameter \'Real.latex(..., nested_scope)\' is no longer used and is '
                          'being deprecated.', DeprecationWarning, 2)
        try:
            from pylatexenc.latexencode import utf8tolatex
        except ImportError:
            raise ImportError("To export latex from qasm "
                              "pylatexenc needs to be installed. Run "
                              "'pip install pylatexenc' before using this "
                              "method.")
        return utf8tolatex(self.value)

    def sym(self, nested_scope=None):
        """Return the correspond symbolic number."""
        del nested_scope  # unused
        return float(self.value)

    def real(self, nested_scope=None):
        """Return the correspond floating point number."""
        del nested_scope  # unused
        return float(self.value.evalf())
