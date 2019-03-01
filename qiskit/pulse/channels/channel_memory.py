# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Memory of channel registers.
"""
from qiskit.providers import BaseBackend


class ChannelMemory:    # TODO: better name?
    """Implement a channel memory."""

    def __init__(self, backend: BaseBackend = None):
        self.drive = None
        self.control = None
        self.measure = None
        self.acquire = None
        self.snapshot = None
        if backend:
            # TODO create default channel registers
            pass
