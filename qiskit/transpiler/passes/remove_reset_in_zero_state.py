# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove reset gate when the qubit is in zero state
"""

from qiskit.circuit import Reset
from qiskit.transpiler.basepasses import TransformationPass


class RemoveResetInZeroState(TransformationPass):
    """Remove reset gate when the qubit is in zero state"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        resets = dag.op_nodes(Reset)
        for reset in resets:
            predecessor = next(dag.predecessors(reset))
            if predecessor.type == 'in':
                dag.remove_op_node(reset)
        return dag
