# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Updates property_set['amount_of_operations'] with the amount of operations in the dag
"""
from qiskit.transpiler._basepasses import AnalysisPass


class CountOperations(AnalysisPass):
    """ Updates property_set['amount_of_operations'] with the amount of operations in the dag
    """

    def run(self, dag):
        self.property_set['amount_of_operations'] = dag.size()
        return dag
