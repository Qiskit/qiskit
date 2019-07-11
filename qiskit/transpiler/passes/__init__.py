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

"""Module containing transpiler pass."""

from .unroller import Unroller
from .cx_cancellation import CXCancellation
from .fixed_point import FixedPoint
from .resource_estimation import ResourceEstimation
from .depth import Depth
from .size import Size
from .width import Width
from .count_ops import CountOps
from .count_ops_longest_path import CountOpsLongestPath
from .num_tensor_factors import NumTensorFactors
from .dag_fixed_point import DAGFixedPoint
from .dag_longest_path import DAGLongestPath
from .merge_adjacent_barriers import MergeAdjacentBarriers
from .optimize_1q_gates import Optimize1qGates
from .decompose import Decompose
from .unroll_3q_or_more import Unroll3qOrMore
from .commutation_analysis import CommutationAnalysis
from .optimize_swap_before_measure import OptimizeSwapBeforeMeasure
from .commutative_cancellation import CommutativeCancellation
from .remove_reset_in_zero_state import RemoveResetInZeroState
from .collect_2q_blocks import Collect2qBlocks
from .consolidate_blocks import ConsolidateBlocks
from .mapping.full_ancilla_allocation import FullAncillaAllocation
from .mapping.enlarge_with_ancilla import EnlargeWithAncilla
from .mapping.apply_layout import ApplyLayout
from .mapping.barrier_before_final_measurements import BarrierBeforeFinalMeasurements
from .mapping.check_map import CheckMap
from .mapping.check_cx_direction import CheckCXDirection
from .mapping.cx_direction import CXDirection
from .mapping.trivial_layout import TrivialLayout
from .mapping.set_layout import SetLayout
from .mapping.dense_layout import DenseLayout
from .mapping.noise_adaptive_layout import NoiseAdaptiveLayout
from .mapping.basic_swap import BasicSwap
from .mapping.lookahead_swap import LookaheadSwap
from .remove_diagonal_gates_before_measure import RemoveDiagonalGatesBeforeMeasure
from .mapping.stochastic_swap import StochasticSwap
