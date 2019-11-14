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

""" An analysis pass for estimating the circuit runtime.
"""
from qiskit.transpiler.basepasses import AnalysisPass
import networkx as nx


class Runtime(AnalysisPass):
    """ An analysis pass for estimating the circuit runtime based on op runtimes.
    """

    def run(self, dag, op_times=None):
        """ Calculate the overall runtime for the DAG longest path by putting the
        operation times as weights on edges.
        """
        weighted_dag = nx.DiGraph()
        for source, target, data in dag.edges():
            try:
                if target.type == 'op':
                    if op_times is None:
                        op_time = 1
                    else:
                        op_time = op_times[target.name]
                else:
                    op_time = 0
                weighted_dag.add_edge(source, target, weight=op_time)
            except KeyError:
                raise KeyError("Could not find {} operation in op_times "
                               "dictionary!".format(target.name))
        runtime = nx.dag_longest_path_length(weighted_dag, weight='weight',
                                             default_weight=0)
        self.property_set['runtime'] = runtime
