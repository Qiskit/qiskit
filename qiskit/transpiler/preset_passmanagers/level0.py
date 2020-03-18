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

"""Pass manager for optimization level 0, providing no explicit optimization.

Level 0 pass manager: no explicit optimization other than mapping to backend.
"""

from qiskit.transpiler.pass_manager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckCXDirection


def level_0_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 0 pass manager: no explicit optimization other than mapping to backend.

    This pass manager applies the user-given initial layout. If none is given, a trivial
    layout consisting of mapping the i-th virtual qubit to the i-th physical qubit is used.
    Any unused physical qubit is allocated as ancilla space.

    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, extra resets are removed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 0 pass manager.
    """
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    seed_transpiler = pass_manager_config.seed_transpiler

    # 1. Use trivial layout if no layout given
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    _choose_layout = TrivialLayout(coupling_map)

    # 2. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 3. Decompose so only 1-qubit and 2-qubit gates remain
    _unroll3q = Unroll3qOrMore()

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set['is_swap_mapped']

    _swap = [BarrierBeforeFinalMeasurements(),
             StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)]

    # 5. Unroll to the basis
    _unroll = Unroller(basis_gates)

    # 6. Fix any bad CX directions
    _direction_check = [CheckCXDirection(coupling_map)]

    def _direction_condition(property_set):
        return not property_set['is_direction_mapped']

    _direction = [CXDirection(coupling_map)]

    # 7. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    # Build pass manager
    pm0 = PassManager()
    if coupling_map:
        pm0.append(_given_layout)
        pm0.append(_choose_layout, condition=_choose_layout_condition)
        pm0.append(_embed)
        pm0.append(_unroll3q)
        pm0.append(_swap_check)
        pm0.append(_swap, condition=_swap_condition)
    pm0.append(_unroll)
    if coupling_map and not coupling_map.is_symmetric:
        pm0.append(_direction_check)
        pm0.append(_direction, condition=_direction_condition)
    pm0.append(_reset)

    return pm0
