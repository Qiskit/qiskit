# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from qiskit.transpiler._basepasses import TransformationPass

class SwapMapper(TransformationPass):

    def __init__(self, coupling_map):
        super().__init__()
        self.coupling_map = coupling_map

    def run(self, dag):
        for layer in dag.layers():
            subdag = layer['graph']
            if not ('cx' in subdag.count_ops() or 'CX' in subdag.count_ops()):
                continue
        return dag