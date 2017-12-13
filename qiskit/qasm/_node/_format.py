# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Node for an OPENQASM file identifier/version statement.
"""
from ._node import Node
import re


class Format(Node):
    """Node for an OPENQASM file identifier/version statement.
    """

    def __init__(self, value):
        """Create the version node."""
        Node.__init__(self, "format", None, None)
        t = re.match('(\w+)\s+(\d+)\.(\d+)', value)
        self.language = t[1]
        self.majorversion = t[2]
        self.minorversion = t[3]

    def version(self):
        return "%s.%s" % (self.majorversion, self.minorversion)

    def qasm(self, prec=15):
        """Return the corresponding format string."""
        return "%s %s;" % (self.language, self.version())
