# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Writes circuit resources to the property set
('size', 'depth', 'width', 'count_ops', 'num_tensor_factors')
"""
from qiskit.transpiler.basepasses import AnalysisPass


class ResourceEstimation(AnalysisPass):
    """ Updates 'size', 'depth', 'width', and 'count_ops' in the property set.
    """

    def run(self, dag):
        self.property_set['size'] = dag.size()
        self.property_set['depth'] = dag.depth()
        self.property_set['width'] = dag.width()
        op_dict = {}
        for node in self.node_nums_in_topological_order():
            nd = self.multi_graph.node[node]
            name = nd["name"]
            if nd["type"] == "op":
                if name not in op_dict:
                    op_dict[name] = 1
                else:
                    op_dict[name] += 1
        self.property_set['count_ops'] = op_dict
        return dag
