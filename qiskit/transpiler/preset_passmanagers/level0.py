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

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import CXDirection
from qiskit.transpiler.passes import SetLayout
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
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckCXDirection
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import TimeUnitAnalysis
from qiskit.transpiler.passes import ALAPSchedule
from qiskit.transpiler.passes import ASAPSchedule
from qiskit.transpiler.passes import Error

from qiskit.transpiler import TranspilerError


def level_0_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 0 pass manager: no explicit optimization other than mapping to backend.

    This pass manager applies the user-given initial layout. If none is given, a trivial
    layout consisting of mapping the i-th virtual qubit to the i-th physical qubit is used.
    Any unused physical qubit is allocated as ancilla space.

    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 0 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or 'trivial'
    routing_method = pass_manager_config.routing_method or 'stochastic'
    translation_method = pass_manager_config.translation_method or 'translator'
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    synthesis_fidelity = pass_manager_config.synthesis_fidelity

    # 1. Choose an initial layout if not set by user (default: trivial layout)
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    if layout_method == 'trivial':
        _choose_layout = TrivialLayout(coupling_map)
    elif layout_method == 'dense':
        _choose_layout = DenseLayout(coupling_map, backend_properties)
    elif layout_method == 'noise_adaptive':
        _choose_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == 'sabre':
        _choose_layout = SabreLayout(coupling_map, max_iterations=1, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    # 2. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 3. Decompose so only 1-qubit and 2-qubit gates remain
    _unroll3q = Unroll3qOrMore()

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set['is_swap_mapped']

    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_method == 'basic':
        _swap += [BasicSwap(coupling_map)]
    elif routing_method == 'stochastic':
        _swap += [StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)]
    elif routing_method == 'lookahead':
        _swap += [LookaheadSwap(coupling_map, search_depth=2, search_width=2)]
    elif routing_method == 'sabre':
        _swap += [SabreSwap(coupling_map, heuristic='basic', seed=seed_transpiler)]
    elif routing_method == 'none':
        _swap += [Error(msg='No routing method selected, but circuit is not routed to device. '
                            'CheckMap Error: {check_map_msg}', action='raise')]
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # 5. Unroll to the basis
    if translation_method == 'unroller':
        _unroll = [Unroller(basis_gates)]
    elif translation_method == 'translator':
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
        _unroll = [UnrollCustomDefinitions(sel, basis_gates),
                   BasisTranslator(sel, basis_gates)]
    elif translation_method == 'synthesis':
        _unroll = [
            Unroll3qOrMore(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates),
            UnitarySynthesis(basis_gates, fidelity=synthesis_fidelity),
        ]
    else:
        raise TranspilerError("Invalid translation method %s." % translation_method)

    # 6. Fix any bad CX directions
    _direction_check = [CheckCXDirection(coupling_map)]

    def _direction_condition(property_set):
        return not property_set['is_direction_mapped']

    _direction = [CXDirection(coupling_map)]

    # 7. Schedule the circuit only when scheduling_method is supplied
    if scheduling_method:
        _scheduling = [TimeUnitAnalysis(instruction_durations)]
        if scheduling_method in {'alap', 'as_late_as_possible'}:
            _scheduling += [ALAPSchedule(instruction_durations)]
        elif scheduling_method in {'asap', 'as_soon_as_possible'}:
            _scheduling += [ASAPSchedule(instruction_durations)]
        else:
            raise TranspilerError("Invalid scheduling method %s." % scheduling_method)

    # Build pass manager
    pm0 = PassManager()
    if coupling_map or initial_layout:
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
    if scheduling_method:
        pm0.append(_scheduling)
    return pm0
