# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Node for an OPENQASM file identifier/version statement.
"""
import re

from ._node import Node


class Format(Node):
    """Node for an OPENQASM file identifier/version statement.
    """

    def __init__(self, value):
        """Create the version node."""
        Node.__init__(self, "format", None, None)
        parts = re.match(r'(\w+)\s+(\d+)\.(\d+)', value)
        self.language = parts.group(1)
        self.majorversion = parts.group(2)
        self.minorversion = parts.group(3)

    def version(self):
        """Return the version."""
        return "%s.%s" % (self.majorversion, self.minorversion)

    def qasm(self, prec=15):
        """Return the corresponding format string."""
        # pylint: disable=unused-argument
        return "%s %s;" % (self.language, self.version())
