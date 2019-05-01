# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Level 0 pass manager:
no optimization, just conforming to basis and coupling map
"""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CheckCXDirection
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import LegacySwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import RemoveResetInZeroState


def level_0_pass_manager(basis_gates, coupling_map, initial_layout, seed_transpiler):
    """
    Level 0 pass manager: no explicit optimization other than mapping to backend.

    Args:
        basis_gates (list[str]): list of basis gate names supported by the target.
        coupling_map (CouplingMap): coupling map to target in mapping.
        initial_layout (Layout or None): initial layout of virtual qubits on physical qubits
        seed_transpiler (int or None): random seed for stochastic passes.

    Returns:
        PassManager: a level 0 pass manager.
    """

    # 1. Use trivial layout if no layout given
    _choose_layout = TrivialLayout(coupling_map)
    _choose_layout_condition = lambda property_set: not property_set['layout']

    # 2. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla()]

    # 3. Unroll to the basis
    _unroll = Unroller(basis_gates)

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)
    _swap = [LegacySwap(coupling_map, trials=20, seed=seed_transpiler),
             Decompose(SwapGate)]
    _swap_condition = lambda property_set: not property_set['is_swap_mapped']

    # 5. Fix any bad CX directions
    # _direction_check = CheckCXDirection(coupling_map)  # TODO
    _direction = [CXDirection(coupling_map)]
    _direction_condition = lambda property_set: not property_set['is_direction_mapped']

    # 6. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    pm0 = PassManager()
    pm0.property_set['layout'] = initial_layout
    pm0.append(_choose_layout, condition=_choose_layout_condition)
    pm0.append(_embed)
    pm0.append(_unroll)
    pm0.append(_swap_check)
    pm0.append(_swap, condition=_swap_condition)
    # pm0.append(_direction_check)
    pm0.append(_direction, condition=_direction_condition)
    pm0.append(_reset)

    return pm0
