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

"""Return the longest path (regarding runtime) in a DAGcircuit as a list of DAGNodes."""

import networkx as nx
from qiskit.transpiler.basepasses import AnalysisPass


class LongestPath(AnalysisPass):
    """ An analysis pass for finding the longest pass based on operation times.
    """

    def __init__(self, op_times=None):
        super().__init__()
        self.op_times = op_times

    def run(self, dag):
        """Return the longest path in a DAGcircuit as a list of DAGNodes."""
        weighted_dag = nx.DiGraph()
        for source, target, _ in dag.edges():
            try:
                if target.type == 'op':
                    if self.op_times is None:
                        op_time = 1
                    else:
                        op_time = self.op_times[target.name]
                    weighted_dag.add_edge(source, target, weight=op_time)
                else:
                    weighted_dag.add_edge(source, target)
            except KeyError:
                raise KeyError("Could not find {} operation in op_times "
                               "dictionary!".format(target.name))
        # networkx has problems handling weight=0
        # so default_weight=1 needs to be ensured, e.g. for out qubits
        longest_path = nx.dag_longest_path(weighted_dag,
                                           weight='weight',
                                           default_weight=1)
        longest_path_length = nx.dag_longest_path_length(weighted_dag,
                                                         weight='weight',
                                                         default_weight=1)
        self.property_set['longest_path'] = longest_path
        self.property_set['longest_path_length'] = longest_path_length
