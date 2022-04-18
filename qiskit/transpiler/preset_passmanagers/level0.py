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
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
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
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.transpiler.passes import ConstrainedReschedule
from qiskit.transpiler.passes import InstructionDurationCheck
from qiskit.transpiler.passes import ValidatePulseGates
from qiskit.transpiler.passes import PulseGates
from qiskit.transpiler.passes import PadDelay
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import ContainsInstruction

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
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "trivial"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target

    # 1. Decompose so only 1-qubit and 2-qubit gates remain
    _unroll3q = [
        # Use unitary synthesis for basis aware decomposition of UnitaryGates
        UnitarySynthesis(
            basis_gates,
            approximation_degree=approximation_degree,
            method=unitary_synthesis_method,
            min_qubits=3,
            plugin_config=unitary_synthesis_plugin_config,
            target=target,
        ),
        Unroll3qOrMore(),
    ]

    # 2. Choose an initial layout if not set by user (default: trivial layout)
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    if layout_method == "trivial":
        _choose_layout = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _choose_layout = DenseLayout(coupling_map, backend_properties, target=target)
    elif layout_method == "noise_adaptive":
        _choose_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _choose_layout = SabreLayout(coupling_map, max_iterations=1, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    # 3. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 4. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_method == "basic":
        _swap += [BasicSwap(coupling_map)]
    elif routing_method == "stochastic":
        _swap += [StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)]
    elif routing_method == "lookahead":
        _swap += [LookaheadSwap(coupling_map, search_depth=2, search_width=2)]
    elif routing_method == "sabre":
        _swap += [SabreSwap(coupling_map, heuristic="basic", seed=seed_transpiler)]
    elif routing_method == "none":
        _swap += [
            Error(
                msg=(
                    "No routing method selected, but circuit is not routed to device. "
                    "CheckMap Error: {check_map_msg}"
                ),
                action="raise",
            )
        ]
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # 5. Unroll to the basis
    if translation_method == "unroller":
        _unroll = [Unroller(basis_gates)]
    elif translation_method == "translator":
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        _unroll = [
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_plugin_config,
                target=target,
            ),
            UnrollCustomDefinitions(sel, basis_gates),
            BasisTranslator(sel, basis_gates, target),
        ]
    elif translation_method == "synthesis":
        _unroll = [
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                min_qubits=3,
                plugin_config=unitary_synthesis_plugin_config,
                target=target,
            ),
            Unroll3qOrMore(),
            Collect2qBlocks(),
            Collect1qRuns(),
            ConsolidateBlocks(basis_gates=basis_gates, target=target),
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                backend_props=backend_properties,
                method=unitary_synthesis_method,
                plugin_config=unitary_synthesis_plugin_config,
                target=target,
            ),
        ]
    else:
        raise TranspilerError("Invalid translation method %s." % translation_method)

    # 6. Fix any bad CX directions
    _direction_check = [CheckGateDirection(coupling_map, target)]

    def _direction_condition(property_set):
        return not property_set["is_direction_mapped"]

    _direction = [GateDirection(coupling_map, target)]

    # Build pass manager
    pm0 = PassManager()
    if coupling_map or initial_layout:
        pm0.append(_given_layout)
        pm0.append(_unroll3q)
        pm0.append(_choose_layout, condition=_choose_layout_condition)
        pm0.append(_embed)
        pm0.append(_swap_check)
        pm0.append(_swap, condition=_swap_condition)
    pm0.append(_unroll)
    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pm0.append(_direction_check)
        pm0.append(_direction, condition=_direction_condition)
        pm0.append(_unroll)
    if inst_map and inst_map.has_custom_gate():
        pm0.append(PulseGates(inst_map=inst_map))

    # 7. Unify all durations (either SI, or convert to dt if known)
    # Schedule the circuit only when scheduling_method is supplied
    # Apply alignment analysis regardless of scheduling for delay validation.
    if scheduling_method:
        # Do scheduling after unit conversion.
        scheduler = {
            "alap": ALAPScheduleAnalysis,
            "as_late_as_possible": ALAPScheduleAnalysis,
            "asap": ASAPScheduleAnalysis,
            "as_soon_as_possible": ASAPScheduleAnalysis,
        }
        pm0.append(TimeUnitConversion(instruction_durations))
        try:
            pm0.append(scheduler[scheduling_method](instruction_durations))
        except KeyError as ex:
            raise TranspilerError("Invalid scheduling method %s." % scheduling_method) from ex
    elif instruction_durations:
        # No scheduling. But do unit conversion for delays.
        def _contains_delay(property_set):
            return property_set["contains_delay"]

        pm0.append(ContainsInstruction("delay"))
        pm0.append(TimeUnitConversion(instruction_durations), condition=_contains_delay)
    if (
        timing_constraints.granularity != 1
        or timing_constraints.min_length != 1
        or timing_constraints.acquire_alignment != 1
        or timing_constraints.pulse_alignment != 1
    ):
        # Run alignment analysis regardless of scheduling.

        def _require_alignment(property_set):
            return property_set["reschedule_required"]

        pm0.append(
            InstructionDurationCheck(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            )
        )
        pm0.append(
            ConstrainedReschedule(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            ),
            condition=_require_alignment,
        )
        pm0.append(
            ValidatePulseGates(
                granularity=timing_constraints.granularity,
                min_length=timing_constraints.min_length,
            )
        )
    if scheduling_method:
        # Call padding pass if circuit is scheduled
        pm0.append(PadDelay())

    return pm0
