# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" TODO
"""
from collections import defaultdict
from qiskit.transpiler._basepasses import AnalysisPass


class FixedPoint(AnalysisPass):
    """ A dummy analysis pass that checks if a property reached a fixed point. The results is saved
        in property_set['fixed_point'][<property>] as a boolean. If a pass instance is set in
        pass_that_updates_the_property, it will be added as a requires dependency.
    """

    def __init__(self, property_to_check, pass_that_updates_the_property=None):
        """
        Args:
            property_to_check (str): The property to check if a fixed point was reached.
            pass_that_updates_the_property (BasePass): The pass instance that updates that property.
        """
        super().__init__()
        self._property = property_to_check
        self._previous_value = None
        if pass_that_updates_the_property:
            self.requires = [pass_that_updates_the_property]

    def run(self, dag):
        if self.property_set['fixed_point'] is None:
            self.property_set['fixed_point'] = defaultdict(lambda: False)

        current_value = self.property_set[self._property]

        if self._previous_value is not None:
            self.property_set['fixed_point'][self._property] = self._previous_value == current_value

        self._previous_value = current_value
