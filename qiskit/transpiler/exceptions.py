# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by the transpiler.
"""
from qiskit.exceptions import QiskitError


class TranspilerError(QiskitError):
    """Exceptions raised during transpilation"""


class TranspilerAccessError(QiskitError):
    """Exception of access error in the transpiler passes."""


class CouplingError(QiskitError):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)


class LayoutError(QiskitError):
    """Errors raised by the layout object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = ' '.join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)
