# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A default passmanager."""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from qiskit.transpiler.passes.unroller import Unroller
from qiskit.transpiler.passes.unroll_3q_or_more import Unroll3qOrMore
from qiskit.transpiler.passes.cx_cancellation import CXCancellation
from qiskit.transpiler.passes.decompose import Decompose
from qiskit.transpiler.passes.optimize_1q_gates import Optimize1qGates
from qiskit.transpiler.passes.fixed_point import FixedPoint
from qiskit.transpiler.passes.depth import Depth
from qiskit.transpiler.passes.remove_reset_in_zero_state import RemoveResetInZeroState
from qiskit.transpiler.passes.mapping.check_map import CheckMap
from qiskit.transpiler.passes.mapping.cx_direction import CXDirection
from qiskit.transpiler.passes.mapping.dense_layout import DenseLayout
from qiskit.transpiler.passes.mapping.trivial_layout import TrivialLayout
from qiskit.transpiler.passes.mapping.legacy_swap import LegacySwap
from qiskit.transpiler.passes.mapping.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.mapping.enlarge_with_ancilla import EnlargeWithAncilla


def default_pass_manager(basis_gates, coupling_map, initial_layout, seed_transpiler):
    """
    The default pass manager that maps to the coupling map.

    Args:
        basis_gates (list[str]): list of basis gate names supported by the target.
        coupling_map (CouplingMap): coupling map to target in mapping.
        initial_layout (Layout or None): initial layout of virtual qubits on physical qubits
        seed_transpiler (int or None): random seed for stochastic passes.

    Returns:
        PassManager: A pass manager to map and optimize.
    """
    pass_manager = PassManager()
    pass_manager.property_set['layout'] = initial_layout

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
    pass_manager.append(LegacySwap(coupling_map, trials=20, seed=seed_transpiler))

    # Expand swaps
    pass_manager.append(Decompose(SwapGate))

    # Change CX directions
    pass_manager.append(CXDirection(coupling_map))

    # simplifications and optimizations
    simplification_passes = [RemoveResetInZeroState(), CXCancellation()]
    if set(basis_gates).issubset(set(['u1', 'u2', 'u3', 'id', 'cx'])):
        # Simplify single qubit gates
        simplification_passes += [Optimize1qGates()]
    pass_manager.append(simplification_passes + [Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])

    # Unroll to the basis
    pass_manager.append(Unroller(basis_gates))

    return pass_manager


def default_pass_manager_simulator(basis_gates):
    """
    The default pass manager without a coupling map.

    Args:
        basis_gates (list[str]): list of basis gate names to unroll to.

    Returns:
        PassManager: A passmanager that just unrolls, without any optimization.
    """
    pass_manager = PassManager()

    pass_manager.append(Unroller(basis_gates))

    pass_manager.append([RemoveResetInZeroState(), Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])

    return pass_manager
