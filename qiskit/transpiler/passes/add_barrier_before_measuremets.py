# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
This pass adds a barrier before the measurements.
"""

from qiskit.extensions.standard.barrier import Barrier
from qiskit.transpiler import TransformationPass

class AddBarrierBeforeMeasuremets(TransformationPass):
    """Adds a barrier before measurements."""

    def __init__(self):
        super().__init__()

    def run(self, dag):
        """Return a circuit with a barrier before last measurments."""
        last_measures = []
        for measures in dag.get_named_nodes('measure'):
            pass
        return dag