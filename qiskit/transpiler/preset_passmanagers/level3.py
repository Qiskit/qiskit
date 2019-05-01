# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Level 3 pass manager:
noise adaptive mapping in addition to heavy optimization based on unitary synthesis
"""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CheckCXDirection
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import LegacySwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks


def level_3_pass_manager(basis_gates, coupling_map, initial_layout,
                         seed_transpiler, backend_properties):
    """
    Level 3 pass manager: heavy optimization by noise adaptive qubit mapping and
    gate cancellation using commutativity rules and unitary synthesis.

    Args:
        basis_gates (list[str]): list of basis gate names supported by the target.
        coupling_map (CouplingMap): coupling map to target in mapping.
        initial_layout (Layout or None): initial layout of virtual qubits on physical qubits
        seed_transpiler (int or None): random seed for stochastic passes.
        backend_properties (BackendProperties): properties of backend containing calibration info

    Returns:
        PassManager: a level 2 pass manager.
    """
    # 1. Layout on good qubits if calibration info available, otherwise on dense links
    _choose_layout = DenseLayout(coupling_map)
    if backend_properties:
        _choose_layout = NoiseAdaptiveLayout(coupling_map)
    _choose_layout_condition = lambda property_set: not property_set['layout']

    # 2. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla()]

    # 3. Unroll to 1q or 2q gates, swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)
    _swap = [Unroll3qOrMore(),
             LegacySwap(coupling_map),
             Decompose(SwapGate)]
    _swap_condition = lambda property_set: not property_set['is_swap_mapped']

    # 4. Fix any bad CX directions
    # _direction_check = CheckCXDirection(coupling_map)  # TODO
    _direction = [CXDirection(coupling_map)]
    _direction_condition = lambda property_set: not property_set['is_direction_mapped']

    # 5. Unroll to the basis
    _unroll = Unroller(basis_gates)

    # 6. 1q rotation merge and commutative cancellation iteratively until no more change in depth
    _depth_check = [Depth(), FixedPoint('depth')]
    _opt = [RemoveResetInZeroState(),
            Collect2qBlocks(), ConsolidateBlocks(),
            Optimize1qGates(), CommutativeCancellation(),
            OptimizeSwapBeforeMeasure(), RemoveDiagonalGatesBeforeMeasure()]
    _opt_control = lambda property_set: not property_set['depth_fixed_point']

    pm3 = PassManager()
    pm3.property_set['layout'] = initial_layout
    pm3.append(_choose_layout, condition=_choose_layout_condition)
    pm3.append(_embed)
    pm3.append(_swap_check)
    pm3.append(_swap, condition=_swap_condition)
    # pm3.append(_direction_check)
    pm3.append(_direction, condition=_direction_condition)
    pm3.append(_unroll)
    pm3.append(_depth_check + _opt, do_while=_opt_control)

    return pm3
