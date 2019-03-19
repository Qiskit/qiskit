# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" An analysis pass for counting operations in a DAG circuit.
"""
from qiskit.transpiler.basepasses import AnalysisPass


class CountOps(AnalysisPass):
    """ An analysis pass for counting operations in a DAG circuit.
    """

    def run(self, dag):
        self.property_set['count_ops'] = dag.count_ops()
