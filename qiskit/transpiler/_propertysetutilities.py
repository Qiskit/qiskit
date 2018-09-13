# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Some utilities for the property set """

from abc import ABC, abstractmethod


class Utility(ABC):
    """
    An Utility is a property set utility that allows to reason about the state of the property set.
    """
    def __init__(self, property_set):
        self.property_set = property_set

    @abstractmethod
    def on_change(self, key, new_value):
        """
        The method is called when the property set is changed, exactly before updating key with
        new_value

        Args:
            key (string): The key to update.
            new_value (any): The new value to store.

        Raises:
            NotImplementedError: This is an abstract method that needs to be implemented
        """
        raise NotImplementedError

    @abstractmethod
    def getter(self):
        """
        This method is called when property_set.class_name is called, where class_name is the
        name of the utility class. It can have more arguments.

        Raises:
            NotImplementedError: This is an abstract method that needs to be implemented
        """
        raise NotImplementedError


class fixed_point(Utility):  # pylint: disable=invalid-name
    """ A property set utility to detect when a property reaches a fixed point."""

    def __init__(self, property_set):
        self.property_fixed_point = {}
        super().__init__(property_set)

    def on_change(self, key, new_value):
        """
        When a property in key set is updated with the same value, this utility turns True
        the property_fixed_point[key] value. False otherwise.

        Args:
            key (string): The key to update.
            new_value (any): The new value to store.
        """
        if self.property_set[key] and new_value:
            self.property_fixed_point[key] = self.property_set[key] == new_value
        else:
            self.property_fixed_point[key] = False

    def getter(self, key):  # pylint: disable=arguments-differ
        return self.property_fixed_point.get(key, False)
