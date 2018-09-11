# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A property set is handle by the PassManager to keep information about the current state of
the  circuit """


class PropertySet:
    """ A dictionary-like object """

    def __init__(self):
        self._properties = {}
        self.utilities = {}

    def __getitem__(self, key):
        return self._properties.get(key, None)

    def __getattr__(self, item):
        if item in self.utilities:
            return self.utilities[item].getter
        raise AttributeError

    def __setitem__(self, key, value):
        for utility in self.utilities:
            self.utilities[utility].on_change(key, value)
        self._properties[key] = value

    def add_utility(self, utility_class):
        """
        Adds an utility for the property set.
        Args:
            utility_class (UtilityClass): The utility class.
        """
        self.utilities[utility_class.__name__] = utility_class(self)
