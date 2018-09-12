# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Some utilities for the property set """

from abc import ABC, abstractmethod

class Utility(ABC):
    def __init__(self, property_set):
        self.property_set = property_set

    @abstractmethod
    def on_change(self, key, new_value):
        raise NotImplementedError

    @abstractmethod
    def getter(self):
        raise NotImplementedError

class fixed_point(Utility): # pylint: disable=invalid-name
    def __init__(self, property_set):
        self.property_fixed_point = {}
        super().__init__(property_set)

    def on_change(self, key, new_value):
        if self.property_set[key] and new_value:
            self.property_fixed_point[key] = self.property_set[key] == new_value
        else:
            self.property_fixed_point[key] = False

    def getter(self, key): # pylint: disable=arguments-differ
        return self.property_fixed_point.get(key, False)
