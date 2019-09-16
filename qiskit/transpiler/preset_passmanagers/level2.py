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

# pylint: disable=unused-variable

"""
Level 2 pass manager:
noise adaptive mapping in addition to commutation-based optimization
"""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckCXDirection


def level_2_pass_manager(transpile_config):
    """
    Level 2 pass manager: medium optimization by noise adaptive qubit mapping and
    gate cancellation using commutativity rules.

    This pass manager applies the user-given initial layout. If none is given, and
    device calibration information is available, the circuit is mapped to the qubits
    with best readouts and to CX gates with highest fidelity. Otherwise, a layout on
    the most densely connected qubits is used.
    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation and redundant
    reset removal are performed.
    Note: in simulators where coupling_map=None, only the unrolling and optimization
    stages are done.

    Args:
        transpile_config (TranspileConfig)

    Returns:
        PassManager: a level 2 pass manager.
    """
    basis_gates = transpile_config.basis_gates
    coupling_map = transpile_config.coupling_map
    initial_layout = transpile_config.initial_layout
    seed_transpiler = transpile_config.seed_transpiler
    backend_properties = transpile_config.backend_properties

    # 1. Unroll to the basis first, to prepare for noise-adaptive layout
    _unroll = Unroller(basis_gates)

    # 2. Layout on good qubits if calibration info available, otherwise on dense links
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    _choose_layout = DenseLayout(coupling_map, backend_properties)

    # 3. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 4. Unroll to 1q or 2q gates, swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set['is_swap_mapped']

    _swap = [BarrierBeforeFinalMeasurements(),
             Unroll3qOrMore(),
             StochasticSwap(coupling_map, trials=20, seed=seed_transpiler),
             Decompose(SwapGate)]

    # 5. Fix any bad CX directions
    _direction_check = [CheckCXDirection(coupling_map)]

    def _direction_condition(property_set):
        return not property_set['is_direction_mapped']

    _direction = [CXDirection(coupling_map)]

    # 6. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    # 7. 1q rotation merge and commutative cancellation iteratively until no more change in depth
    _depth_check = [Depth(), FixedPoint('depth')]

    def _opt_control(property_set):
        return not property_set['depth_fixed_point']

    _opt = [Optimize1qGates(), CommutativeCancellation()]

    pm2 = PassManager()
    pm2.append(_unroll)
    if coupling_map:
        pm2.append(_given_layout)
        pm2.append(_choose_layout, condition=_choose_layout_condition)
        pm2.append(_embed)
        pm2.append(_swap_check)
        pm2.append(_swap, condition=_swap_condition)
        if not coupling_map.is_symmetric:
            pm2.append(_direction_check)
            pm2.append(_direction, condition=_direction_condition)
    pm2.append(_reset)
    pm2.append(_depth_check + _opt, do_while=_opt_control)

    return pm2
