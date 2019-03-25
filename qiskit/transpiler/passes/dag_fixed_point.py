# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Detects when the DAG reached a fixed point (it's not modified anymore)
"""
from copy import deepcopy

from qiskit.transpiler.basepasses import AnalysisPass


class DAGFixedPoint(AnalysisPass):
    """ A dummy analysis pass that checks if the DAG a fixed point. The results is saved
        in property_set['dag_fixed_point'] as a boolean.
    """

    def run(self, dag):
        if self.property_set['_dag_fixed_point_previous_dag'] is None:
            self.property_set['dag_fixed_point'] = False
        else:
            fixed_point_reached = self.property_set['_dag_fixed_point_previous_dag'] == dag
            self.property_set['dag_fixed_point'] = fixed_point_reached

        self.property_set['_dag_fixed_point_previous_dag'] = deepcopy(dag)
