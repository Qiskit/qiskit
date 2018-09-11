# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Some utilities for the property set """

class Utility():
    def __init__(self, property_set):
        raise NotImplementedError

    def on_change(self, key, new_value):
        raise NotImplementedError

    def getter(self):
        raise NotImplementedError

class fixed_point(Utility):
    def __init__(self, property_set):
        self.property_set = property_set
        self.property_fixed_point = {}

    def on_change(self, key, new_value):
        if self.property_set[key] and new_value:
            self.property_fixed_point[key] = self.property_set[key] == new_value
        else:
            self.property_fixed_point[key] = False

    def getter(self, key):
        return self.property_fixed_point.get(key, False)
