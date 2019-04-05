# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" An analysis pass for automatically running Depth(), Width(), Size(), CountOps(), and
Tensor_Factor()
"""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.depth import Depth
from qiskit.transpiler.passes.width import Width
from qiskit.transpiler.passes.size import Size
from qiskit.transpiler.passes.count_ops import CountOps
from qiskit.transpiler.passes.num_tensor_factors import NumTensorFactors


class ResourceEstimation(AnalysisPass):
    """ Requires Depth(), Width(), Size(), CountOps(), and NumTensorFactors().
    """

    def __init__(self):
        super().__init__()
        self.requires += [Depth(), Width(), Size(), CountOps(), NumTensorFactors()]

    def run(self, _):
        pass
