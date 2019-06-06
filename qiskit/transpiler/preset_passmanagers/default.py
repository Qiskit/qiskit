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

"""A default passmanager."""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import LegacySwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla


def default_pass_manager(transpile_config):
    """
    The default pass manager that maps to the coupling map.

    Args:
        transpile_config (TranspileConfig)

    Returns:
        PassManager: A pass manager to map and optimize.
    """
    basis_gates = transpile_config.basis_gates
    coupling_map = transpile_config.coupling_map
    initial_layout = transpile_config.initial_layout
    seed_transpiler = transpile_config.seed_transpiler
    pass_manager = PassManager()
    pass_manager.append(SetLayout(initial_layout))
    pass_manager.append(Unroller(basis_gates))

    # Use the trivial layout if no layout is found
    pass_manager.append(TrivialLayout(coupling_map),
                        condition=lambda property_set: not property_set['layout'])

    # if the circuit and layout already satisfy the coupling_constraints, use that layout
    # otherwise layout on the most densely connected physical qubit subset
    pass_manager.append(CheckMap(coupling_map))
    pass_manager.append(DenseLayout(coupling_map),
                        condition=lambda property_set: not property_set['is_swap_mapped'])

    # Extend the the dag/layout with ancillas using the full coupling map
    pass_manager.append(FullAncillaAllocation(coupling_map))
    pass_manager.append(EnlargeWithAncilla())

    # Circuit must only contain 1- or 2-qubit interactions for swapper to work
    pass_manager.append(Unroll3qOrMore())

    # Swap mapper
    pass_manager.append(BarrierBeforeFinalMeasurements())
    pass_manager.append(LegacySwap(coupling_map, trials=20, seed=seed_transpiler))

    # Expand swaps
    pass_manager.append(Decompose(SwapGate))

    # Change CX directions
    pass_manager.append(CXDirection(coupling_map),
                        condition=lambda property_set: (not coupling_map.is_symmetric and
                                                        not property_set['is_direction_mapped']))

    # Simplify single qubit gates and CXs
    simplification_passes = [Optimize1qGates(), CXCancellation(), RemoveResetInZeroState()]

    pass_manager.append(simplification_passes + [Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])

    return pass_manager


def default_pass_manager_simulator(transpile_config):
    """
    The default pass manager without a coupling map.

    Args:
        transpile_config (TranspileConfig)

    Returns:
        PassManager: A passmanager that just unrolls, without any optimization.
    """
    basis_gates = transpile_config.basis_gates

    pass_manager = PassManager()
    pass_manager.append(Unroller(basis_gates))
    pass_manager.append([RemoveResetInZeroState(), Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])

    return pass_manager
