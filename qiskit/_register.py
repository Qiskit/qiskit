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
Base register reference object.
"""
import re
from ._qiskiterror import QISKitError


class Register(object):
    """Implement a generic register."""

    def __init__(self, name, size):
        """Create a new generic register."""
        test = re.compile('[a-z][a-zA-Z0-9_]*')
        if test.match(name) is None:
            raise QISKitError("invalid OPENQASM register name")
        self.name = name
        self.size = size
        if size <= 0:
            raise QISKitError("register size must be positive")

    def __str__(self):
        """Return a string representing the register."""
        return "Register(%s,%d)" % (self.name, self.size)

    def __len__(self):
        """Return register size"""
        return self.size

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.size:
            raise QISKitError("register index out of range")

    def __getitem__(self, key):
        """Return tuple (self, key) if key is valid."""
        if not isinstance(key, int):
            raise QISKitError("expected integer index into register")
        self.check_range(key)
        return (self, key)
