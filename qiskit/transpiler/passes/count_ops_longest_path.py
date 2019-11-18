# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Count the operations on the longest path in a DAGcircuit."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.longest_path import LongestPath


class CountOpsLongestPath(AnalysisPass):
    """Count the operations on the longest path in a DAGcircuit."""

    def __init__(self, op_times=None):
        super().__init__()
        self.requires.append(LongestPath(op_times))

    def run(self, dag):
        """Run the CountOpsLongestPath pass on `dag`."""
        op_dict = {}
        path = self.property_set['longest_path']
        path = path[1:]     # remove qubit at beginning
        for node in path:
            name = node.name
            if name not in op_dict:
                op_dict[name] = 1
            else:
                op_dict[name] += 1
        self.property_set['count_ops_longest_path'] = op_dict
