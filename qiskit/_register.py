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
import itertools
import warnings

from ._qiskiterror import QISKitError, QISKitIndexError

logger = logging.getLogger(__name__)


class Register(object):
    """Implement a generic register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = 'reg'

    def __init__(self, size, name=None):
        """Create a new generic register.

        .. deprecated:: 0.5
            The `name` parameter will be optional in upcoming versions (>0.5.0)
            and the order of the parameters will change (`size`, `name`)
            instead of (`name`, `size`).
        """

        if isinstance(size, str):
            warnings.warn(
                "name will be optional in upcoming versions (>0.5.0) "
                "and order will be size, name.", DeprecationWarning)
            name_temp = size
            size = name
            name = name_temp

        if name is None:
            name = '%s%i' % (self.prefix, next(self.instances_counter))

        if not isinstance(name, str):
            raise QISKitError("The circuit name should be a string "
                              "(or None for autogenerate a name).")

        test = re.compile('[a-z][a-zA-Z0-9_]*')
        if test.match(name) is None:
            raise QISKitError("%s is an invalid OPENQASM register name." % name)

        self.name = name
        self.size = size
        if size <= 0:
            raise QISKitError("register size must be positive")

    def __repr__(self):
        """Return the official string representing the register."""
        return "%s(%d, '%s')" % (self.__class__.__qualname__,
                                 self.size, self.name)

    def __len__(self):
        """Return register size"""
        return self.size

    def check_range(self, j):
        """Check that j is a valid index into self."""
        if j < 0 or j >= self.size:
            raise QISKitIndexError("register index out of range")

    def __getitem__(self, key):
        """
        Arg:
            key (int): index of the bit/qubit to be retrieved.

        Returns:
            tuple[Register, int]: a tuple in the form `(self, key)`.

        Raises:
            QISKitError: if the `key` is not an integer.
            QISKitIndexError: if the `key` is not in the range
                `(0, self.size)`.
        """
        if not isinstance(key, int):
            raise QISKitError("expected integer index into register")
        self.check_range(key)
        return self, key

    def __iter__(self):
        """
        Returns:
            iterator: an iterator over the bits/qubits of the register, in the
                form `tuple (Register, int)`.
        """
        return zip([self]*self.size, range(self.size))
