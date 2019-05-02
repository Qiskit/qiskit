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
Level 1 pass manager:
mapping in addition to light optimization via adjacent gate collapse
"""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import LegacySwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGates


def level_1_pass_manager(transpile_config):
    """
    Level 1 pass manager: light optimization by simple adjacent gate collapsing

    This pass manager applies the user-given initial layout. If none is given, and a trivial
    layout (i-th virtual -> i-th physical) makes the circuit fit the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph, and swaps
    are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.
    Note: in simulators where coupling_map=None, only the unrolling and optimization
    stages are done.

    Args:
        transpile_config (TranspileConfig)

    Returns:
        PassManager: a level 1 pass manager.
    """
    basis_gates = transpile_config.basis_gates
    coupling_map = transpile_config.coupling_map
    initial_layout = transpile_config.initial_layout
    seed_transpiler = transpile_config.seed_transpiler

    # 1. Use trivial layout if no layout given
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    _choose_layout = TrivialLayout(coupling_map)

    # 2. Use a better layout on densely connected qubits, if circuit needs swaps
    _layout_check = CheckMap(coupling_map)

    def _improve_layout_condition(property_set):
        return not property_set['is_swap_mapped']

    _improve_layout = DenseLayout(coupling_map)

    # 2. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla()]

    # 3. Unroll to the basis
    _unroll = Unroller(basis_gates)

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set['is_swap_mapped']

    _swap = [BarrierBeforeFinalMeasurements(),
             LegacySwap(coupling_map, trials=20, seed=seed_transpiler),
             Decompose(SwapGate)]

    # 5. Fix any bad CX directions
    # _direction_check = CheckCXDirection(coupling_map)  # TODO
    def _direction_condition(property_set):
        return not property_set['is_direction_mapped']

    _direction = [CXDirection(coupling_map)]

    # 6. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    # 7. Merge 1q rotations and cancel CNOT gates iteratively until no more change in depth
    _depth_check = [Depth(), FixedPoint('depth')]

    def _opt_control(property_set):
        return not property_set['depth_fixed_point']

    _opt = [Optimize1qGates(), CXCancellation()]

    pm1 = PassManager()
    if coupling_map:
        pm1.append(_given_layout)
        pm1.append(_choose_layout, condition=_choose_layout_condition)
        pm1.append(_layout_check)
        pm1.append(_improve_layout, condition=_improve_layout_condition)
        pm1.append(_embed)
    pm1.append(_unroll)
    if coupling_map:
        pm1.append(_swap_check)
        pm1.append(_swap, condition=_swap_condition)
        # pm1.append(_direction_check)  # TODO
        pm1.append(_direction, condition=_direction_condition)
    pm1.append(_reset)
    pm1.append(_depth_check + _opt, do_while=_opt_control)

    return pm1
