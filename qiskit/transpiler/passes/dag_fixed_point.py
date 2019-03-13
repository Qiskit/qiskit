# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Detects when the DAG reached a fixed point (it's not modified anymore)
"""
from collections import defaultdict
from qiskit.transpiler.basepasses import AnalysisPass
from copy import copy

class DAGFixedPoint(AnalysisPass):
    """ A dummy analysis pass that checks if the DAG a fixed point. The results is saved
        in property_set['dag_fixed_point'] as a boolean.
    """

    def __init__(self):
        super().__init__()
        self._previous_value = None

    def run(self, dag):
        if self.property_set['dag_fixed_point'] is None:
            self.property_set['dag_fixed_point'] = defaultdict(lambda: False)

        if self._previous_value is not None:
            self.property_set['dag_fixed_point'] = self._previous_value == dag

        self._previous_value = copy(dag)
