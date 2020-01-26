# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing circuit analysis passes."""

from .depth import Depth
from .width import Width
from .size import Size
from .dag_deepest_path import DAGDeepestPath
from .dag_longest_path import DAGLongestPath
from .runtime import Runtime
from .count_ops import CountOps
from .count_ops_deepest_path import CountOpsDeepestPath
from .count_ops_longest_path import CountOpsLongestPath
from .num_tensor_factors import NumTensorFactors
from .num_qubits import NumQubits
from .resource_estimation import ResourceEstimation