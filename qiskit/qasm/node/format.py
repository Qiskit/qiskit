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

from .node import Node


class Format(Node):
    """Node for an OPENQASM file identifier/version statement."""

    def __init__(self, value):
        """Create the version node."""
        super().__init__("format", None, None)
        parts = re.match(r"(\w+)\s+(\d+)(\.(\d+))?", value)
        self.language = parts.group(1)
        self.majorversion = parts.group(2)
        self.minorversion = parts.group(4) if parts.group(4) is not None else "0"

    def version(self):
        """Return the version."""
        return f"{self.majorversion}.{self.minorversion}"

    def qasm(self):
        """Return the corresponding format string."""
        return f"{self.language} {self.version()};"
