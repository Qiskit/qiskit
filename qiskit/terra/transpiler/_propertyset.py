# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A property set is maintained by the PassManager to keep information
about the current state of the circuit """


class PropertySet:
    """ A dictionary-like object """

    def __init__(self):
        self._properties = {}

    def __getitem__(self, key):
        return self._properties.get(key, None)

    def __setitem__(self, key, value):
        self._properties[key] = value
