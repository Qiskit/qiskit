# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Module containing transpiler pass."""

from .unroller import Unroller
from .cx_cancellation import CXCancellation
from .fixed_point import FixedPoint
from .resource_estimation import ResourceEstimation
from .depth import Depth
from .size import Size
from .width import Width
from .count_ops import CountOps
from .num_tensor_factors import NumTensorFactors
from .dag_fixed_point import DAGFixedPoint
from .merge_adjacent_barriers import MergeAdjacentBarriers
from .optimize_1q_gates import Optimize1qGates
from .decompose import Decompose
from .unroll_3q_or_more import Unroll3qOrMore
from .commutation_analysis import CommutationAnalysis
from .optimize_swap_before_measure import OptimizeSwapBeforeMeasure
from .commutative_cancellation import CommutativeCancellation
from .remove_reset_in_zero_state import RemoveResetInZeroState
from .collect_2q_blocks import Collect2qBlocks
from .mapping.barrier_before_final_measurements import BarrierBeforeFinalMeasurements
from .mapping.check_map import CheckMap
from .mapping.check_cnot_direction import CheckCnotDirection
from .mapping.cx_direction import CXDirection
from .mapping.trivial_layout import TrivialLayout
from .mapping.dense_layout import DenseLayout
from .mapping.extend_layout import ExtendLayout
from .mapping.basic_swap import BasicSwap
from .mapping.lookahead_swap import LookaheadSwap
from .mapping.stochastic_swap import StochasticSwap
from .mapping.legacy_swap import LegacySwap
from .mapping.enlarge_with_ancilla import EnlargeWithAncilla
from .mapping.noise_adaptive_layout import NoiseAdaptiveLayout
