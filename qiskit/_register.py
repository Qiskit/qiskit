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
import logging
import warnings
import random
import string

from ._qiskiterror import QISKitError

logger = logging.getLogger(__name__)


class Register(object):
    """Implement a generic register."""
    # Class variable keeping track of number of instances, useful for auto naming.
    # Auto naming prefix
    instances = 0
    prefix = 'reg'

    def __init__(self, size, name=None):
        """Create a new generic register."""
        self._increment_instances()
        if isinstance(size, str):
            warnings.warn(
                "name will be optional in upcoming versions (>0.5.0). "
                "and order will be size, name.", DeprecationWarning)
            name_temp = size
            size = name
            name = name_temp
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())
        self.name = name
        self.size = size
        self._openqasm_name = None
        if size <= 0:
            raise QISKitError("register size must be positive")

    @classmethod
    def _increment_instances(cls):
        cls.instances += 1

    @classmethod
    def cls_instances(cls):
        return cls.instances

    @classmethod
    def cls_prefix(cls):
        return cls.prefix

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

    @property
    def openqasm_name(self):
        """Converts names to strings that are OpenQASM 2.0 complain."""
        if self._openqasm_name is not None:
            return self._openqasm_name
        test = re.compile('[a-z][a-zA-Z0-9_]*')
        if test.match(str(self.name)) is None:
            oq_name = "id%i" % id(self.name)
            logger.info("The name %s is an invalid OpenQASM register name."
                        "Coverting it to %s", self.name, oq_name)
            self._openqasm_name = oq_name
            return oq_name
        self._openqasm_name = self.name
        return str(self.name)

    def __getitem__(self, key):
        """Return tuple (self, key) if key is valid."""
        if not isinstance(key, int):
            raise QISKitError("expected integer index into register")
        self.check_range(key)
        return (self, key)
