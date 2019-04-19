# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""A default passmanager."""

from qiskit.transpiler.passmanager import PassManager
from qiskit.extensions.standard import SwapGate

from ..passes.unroller import Unroller
from ..passes.cx_cancellation import CXCancellation
from ..passes.decompose import Decompose
from ..passes.optimize_1q_gates import Optimize1qGates
from ..passes.fixed_point import FixedPoint
from ..passes.depth import Depth
from ..passes.mapping.check_map import CheckMap
from ..passes.mapping.cx_direction import CXDirection
from ..passes.mapping.dense_layout import DenseLayout
from ..passes.mapping.trivial_layout import TrivialLayout
from ..passes.mapping.legacy_swap import LegacySwap
from ..passes.mapping.enlarge_with_ancilla import EnlargeWithAncilla
from ..passes.mapping.extend_layout import ExtendLayout


def default_pass_manager(basis_gates, coupling_map, initial_layout, seed_mapper):
    """
    The default pass manager that maps to the coupling map.

    Args:
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        initial_layout (Layout or None): If None, trivial layout will be chosen.
        coupling_map (CouplingMap): coupling map (perhaps custom) to target
            in mapping.
        seed_mapper (int or None): random seed for the swap_mapper.

    Returns:
        PassManager: A pass manager to map and optimize.
    """
    pass_manager = PassManager()
    pass_manager.property_set['layout'] = initial_layout

    pass_manager.append(Unroller(basis_gates))

    # Use the trivial layout if no layouto is found
    pass_manager.append(TrivialLayout(coupling_map),
                        condition=lambda property_set: not property_set['layout'])

    # if the circuit and layout already satisfy the coupling_constraints, use that layout
    # otherwise layout on the most densely connected physical qubit subset
    pass_manager.append(CheckMap(coupling_map))
    pass_manager.append(DenseLayout(coupling_map),
                        condition=lambda property_set: not property_set['is_swap_mapped'])

    # Extend and enlarge the the dag/layout with ancillas using the full coupling map
    pass_manager.append(ExtendLayout(coupling_map))
    pass_manager.append(EnlargeWithAncilla())

    # Swap mapper
    pass_manager.append(LegacySwap(coupling_map, trials=20, seed=seed_mapper))

    # Expand swaps
    pass_manager.append(Decompose(SwapGate))

    # Change CX directions
    pass_manager.append(CXDirection(coupling_map))

    # Unroll to the basis
    pass_manager.append(Unroller(['u1', 'u2', 'u3', 'id', 'cx']))

    # Simplify single qubit gates and CXs
    pass_manager.append([Optimize1qGates(), CXCancellation(), Depth(), FixedPoint('depth')],
                        do_while=lambda property_set: not property_set['depth_fixed_point'])
    return pass_manager


def default_pass_manager_simulator(basis_gates):
    """
    The default pass manager without a coupling map.

    Args:
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']

    Returns:
        PassManager: A passmanager without any optimization
    """
    pass_manager = PassManager()
    pass_manager.append(Unroller(basis_gates))

    return pass_manager
