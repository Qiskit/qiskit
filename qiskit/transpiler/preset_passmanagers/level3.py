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

"""Pass manager for optimization level 3, providing heavy optimization.

Level 3 pass manager: heavy optimization by noise adaptive qubit mapping and
gate cancellation using commutativity rules and unitary synthesis.
"""


from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import CSPLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
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
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckCXDirection

from qiskit.transpiler import TranspilerError


def level_3_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 3 pass manager: heavy optimization by noise adaptive qubit mapping and
    gate cancellation using commutativity rules and unitary synthesis.

    This pass manager applies the user-given initial layout. If none is given, a search
    for a perfect layout (i.e. one that satisfies all 2-qubit interactions) is conducted.
    If no such layout is found, and device calibration information is available, the
    circuit is mapped to the qubits with best readouts and to CX gates with highest fidelity.

    The pass manager then transforms the circuit to match the coupling constraints.
    It is then unrolled to the basis, and any flipped cx directions are fixed.
    Finally, optimizations in the form of commutative gate cancellation, resynthesis
    of two-qubit unitary blocks, and redundant reset removal are performed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 3 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or 'dense'
    routing_method = pass_manager_config.routing_method or 'stochastic'
    translation_method = pass_manager_config.translation_method or 'translator'
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties

    # 1. Unroll to 1q or 2q gates
    _unroll3q = Unroll3qOrMore()

    # 2. Layout on good qubits if calibration info available, otherwise on dense links
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    _choose_layout_1 = CSPLayout(coupling_map, call_limit=10000, time_limit=60)
    if layout_method == 'trivial':
        _choose_layout_2 = TrivialLayout(coupling_map)
    elif layout_method == 'dense':
        _choose_layout_2 = DenseLayout(coupling_map, backend_properties)
    elif layout_method == 'noise_adaptive':
        _choose_layout_2 = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == 'sabre':
        _choose_layout_2 = SabreLayout(coupling_map, max_iterations=4, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    # 3. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set['is_swap_mapped']

    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_method == 'basic':
        _swap += [BasicSwap(coupling_map)]
    elif routing_method == 'stochastic':
        _swap += [StochasticSwap(coupling_map, trials=200, seed=seed_transpiler)]
    elif routing_method == 'lookahead':
        _swap += [LookaheadSwap(coupling_map, search_depth=5, search_width=6)]
    elif routing_method == 'sabre':
        _swap += [SabreSwap(coupling_map, heuristic='decay', seed=seed_transpiler)]
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # 5. Unroll to the basis
    if translation_method == 'unroller':
        _unroll = [Unroller(basis_gates)]
    elif translation_method == 'translator':
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
        _unroll = [UnrollCustomDefinitions(sel, basis_gates),
                   BasisTranslator(sel, basis_gates)]
    else:
        raise TranspilerError("Invalid translation method %s." % translation_method)

    # 6. Fix any CX direction mismatch
    _direction_check = [CheckCXDirection(coupling_map)]

    def _direction_condition(property_set):
        return not property_set['is_direction_mapped']

    _direction = [CXDirection(coupling_map)]

    # 8. Optimize iteratively until no more change in depth. Removes useless gates
    # after reset and before measure, commutes gates and optimizes continguous blocks.
    _depth_check = [Depth(), FixedPoint('depth')]

    def _opt_control(property_set):
        return not property_set['depth_fixed_point']

    _reset = [RemoveResetInZeroState()]

    _meas = [OptimizeSwapBeforeMeasure(), RemoveDiagonalGatesBeforeMeasure()]

    _opt = [
        Collect2qBlocks(),
        ConsolidateBlocks(basis_gates=basis_gates),
        UnitarySynthesis(basis_gates),
        Optimize1qGates(basis_gates),
        CommutativeCancellation(),
    ]

    # Build pass manager
    pm3 = PassManager()
    pm3.append(_unroll3q)
    pm3.append(_reset + _meas)
    if coupling_map:
        pm3.append(_given_layout)
        pm3.append(_choose_layout_1, condition=_choose_layout_condition)
        pm3.append(_choose_layout_2, condition=_choose_layout_condition)
        pm3.append(_embed)
        pm3.append(_swap_check)
        pm3.append(_swap, condition=_swap_condition)
    pm3.append(_depth_check + _opt + _unroll, do_while=_opt_control)
    if coupling_map and not coupling_map.is_symmetric:
        pm3.append(_direction_check)
        pm3.append(_direction, condition=_direction_condition)
    pm3.append(_reset)

    return pm3
