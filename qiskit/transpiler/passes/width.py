# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" An analysis pass for calculating the width of a DAG circuit.
"""
from qiskit.transpiler.basepasses import AnalysisPass


class Width(AnalysisPass):
    """ An analysis pass for calculating the width of a DAG circuit.
    """

    def run(self, dag):
        self.property_set['width'] = dag.width()
