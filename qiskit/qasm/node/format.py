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

"""Node for an OPENQASM file identifier/version statement."""

import re
import warnings

from .node import Node


class Format(Node):
    """Node for an OPENQASM file identifier/version statement."""

    def __init__(self, value):
        """Create the version node."""
        super().__init__("format", None, None)
        parts = re.match(r"(\w+)\s+(\d+)\.(\d+)", value)
        self.language = parts.group(1)
        self.majorversion = parts.group(2)
        self.minorversion = parts.group(3)

    def version(self):
        """Return the version."""
        return f"{self.majorversion}.{self.minorversion}"

    def qasm(self, prec=None):
        """Return the corresponding format string."""
        if prec is not None:
            warnings.warn(
                "Parameter 'Format.qasm(..., prec)' is no longer used and is being " "deprecated.",
                DeprecationWarning,
                2,
            )
        return f"{self.language} {self.version()};"
