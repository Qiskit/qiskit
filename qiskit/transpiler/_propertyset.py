# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A property set is handle by the PassManager to keep information about the current state of
the  circuit """

class PropertySet:
    def __init__(self):
        self._properties = {}

        # This map _properties -> Bool
        # If your the same value was previously there (with exception of None), then you reached a
        # fixed point and you set this map as True
        self._property_fixed_point = {}

    def __getitem__(self, key):
        return self._properties.get(key, None)

    def __setitem__(self, key, value):
        if value:
            if self._properties.get(key, None):
                self._property_fixed_point[key] = self._properties[key] == value
            self._properties[key] = value

    def fixed_point(self, property):
        """Returns true when property reaches a fixed point"""
        return self._property_fixed_point.get(property, False)