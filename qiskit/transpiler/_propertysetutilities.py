# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Some utilities for the property set. Before a property is set, registered utilities (via
    add_utility) are called with 3 arguments:
        property_set (PropertySet): The property set that will be updated.
        key (string): The key to update.
        new_value (any): The new value to store.
    The return value of the utility is ignored. The utility can make changes in the property set.
"""

from collections import defaultdict

def fixed_point(property_set, key, new_value):
    """
    A property set utility to detect when a property reaches a fixed point.

    Args:
        property_set:
        key:
        new_value:

    Returns:
        None

    """
    if property_set['fixed_point'] is None:
        property_set.setitem('fixed_point', defaultdict(lambda : False))

    if new_value is None:
        property_set['fixed_point'][key] = False
    else:
        property_set['fixed_point'][key] = property_set[key] == new_value