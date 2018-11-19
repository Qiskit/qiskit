# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
A two-ways dict that represent a layout.

Layout is the relation between logical qubits and physical qubits.
Local qubits are tuples (eg, `('qr',2)`.
Physical qubits are numbers.
"""

class Layout(dict):
    """ Two-ways dict to represent a Layout."""
    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2

    def add_logical(self, logical, physical=None):
        if physical is None:
            physical = len(self)
        self[logical] = physical