# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" The FixedPoint pass detects fixed points in properties.
"""
from copy import deepcopy

from qiskit.transpiler.basepasses import AnalysisPass


class FixedPoint(AnalysisPass):
    """ A dummy analysis pass that checks if a property reached a fixed point. The results is saved
        in property_set['<property>_fixed_point'] as a boolean.
    """

    def __init__(self, property_to_check):
        """
        Args:
            property_to_check (str): The property to check if a fixed point was reached.
        """
        super().__init__()
        self._property = property_to_check

    def run(self, dag):
        current_value = self.property_set[self._property]
        fixed_point_previous_property = '_fixed_point_previous_%s' % self._property

        if self.property_set[fixed_point_previous_property] is None:
            self.property_set['%s_fixed_point' % self._property] = False
        else:
            fixed_point_reached = self.property_set[fixed_point_previous_property] == current_value
            self.property_set['%s_fixed_point' % self._property] = fixed_point_reached

        self.property_set[fixed_point_previous_property] = deepcopy(current_value)
