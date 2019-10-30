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

""" An analysis pass for automatically running Depth(), Width(), Size(), CountOps(), and
Tensor_Factor()
"""
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.depth import Depth
from qiskit.transpiler.passes.width import Width
from qiskit.transpiler.passes.size import Size
from qiskit.transpiler.passes.count_ops import CountOps
from qiskit.transpiler.passes.num_tensor_factors import NumTensorFactors
from qiskit.transpiler.passes.num_qubits import NumQubits


class ResourceEstimation(AnalysisPass):
    """ Requires Depth(), Width(), Size(), CountOps(), NumTensorFactors(), and NumQubits.
    """

    def __init__(self):
        super().__init__()
        self.requires += [Depth(), Width(), Size(), CountOps(), NumTensorFactors(), NumQubits()]

    def run(self, _):
        pass
