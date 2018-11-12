# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit

class SwapMapper(TransformationPass):

    def __init__(self, coupling_map):
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        # new_dag = DAGCircuit.copy_without_gates(dag)
        new_dag = DAGCircuit()
        for layer in dag.serial_layers():
            subdag = layer['graph']
            cxs = subdag.get_cnot_nodes()
            if not cxs:
                # Trivial layer, there is no entanglement in this layer, just leave it like this.
                new_dag.add_dag_at_the_end(subdag)
                continue
            for cx in subdag.get_cnot_nodes():
                dist = self._distance_between_qargs(cx['qargs'])
                if dist == 1:
                    # The CXs are already together, no need to change anything.
                    new_dag.add_dag_at_the_end(subdag)
                    continue
                else:
                    pass #TODO
        return new_dag

    def _distance_between_qargs(self, qargs):
        return self.coupling_map.dist[qargs[0]][qargs[1]]