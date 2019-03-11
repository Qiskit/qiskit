# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" Updates 'size', 'depth', and 'width' in the property set.
"""
from qiskit.transpiler.basepasses import AnalysisPass


class ResourceEstimation(AnalysisPass):
    """ Updates 'size', 'depth', and 'width' in the property set.
    """

    def run(self, dag):
        self.property_set['size'] = dag.size()
        self.property_set['depth'] = dag.depth()
        self.property_set['width'] = dag.width()
        return dag
