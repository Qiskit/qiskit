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

"""Module containing transpiler optimization passes."""

from .optimize_1q_gates import Optimize1qGates
from .optimize_1q_decomposition import Optimize1qGatesDecomposition
from .collect_2q_blocks import Collect2qBlocks
from .consolidate_blocks import ConsolidateBlocks
from .commutation_analysis import CommutationAnalysis
from .commutative_cancellation import CommutativeCancellation
from .cx_cancellation import CXCancellation
from .optimize_swap_before_measure import OptimizeSwapBeforeMeasure
from .remove_reset_in_zero_state import RemoveResetInZeroState
from .remove_diagonal_gates_before_measure import RemoveDiagonalGatesBeforeMeasure
from .crosstalk_adaptive_schedule import CrosstalkAdaptiveSchedule
from .hoare_opt import HoareOptimizer
from .template_optimization import TemplateOptimization
