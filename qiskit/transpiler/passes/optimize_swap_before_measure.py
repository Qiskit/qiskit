# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Transpiler pass to remove the swaps in front of measurments by moving the reading qubit
 of the measure intruction.
"""

from qiskit.circuit import Measure
from qiskit.extensions.standard import SwapGate
from qiskit.transpiler._basepasses import TransformationPass


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed measurments (and adapts the measurement"""

    def run(self, dag):
        """Return a new circuit that has been optimized."""
        swap_runs = dag.collect_runs(['swap'])
        for swap in swap_runs:
            # after_swap = dag.successors(swap[-1])
            print(swap)
        return dag

