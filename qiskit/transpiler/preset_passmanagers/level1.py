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

"""Pass manager for optimization level 1, providing light optimization.

Level 1 pass manager: light optimization by simple adjacent gate collapsing.
"""

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager

from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import CXCancellation
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import Layout2qDistance
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import Size
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import Collect2qBlocks
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
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

from qiskit.transpiler import TranspilerError


def level_1_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    """Level 1 pass manager: light optimization by simple adjacent gate collapsing.

    This pass manager applies the user-given initial layout. If none is given,
    and a trivial layout (i-th virtual -> i-th physical) makes the circuit fit
    the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph,
    and swaps are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.

    Note:
        In simulators where ``coupling_map=None``, only the unrolling and
        optimization stages are done.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 1 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"
    translation_method = pass_manager_config.translation_method or "translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    target = pass_manager_config.target

    # 1. Use trivial layout if no layout given if that isn't perfect use vf2 layout
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    def _trivial_not_perfect(property_set):
        # Verify that a trivial layout is perfect. If trivial_layout_score > 0
        # the layout is not perfect. The layout is unconditionally set by trivial
        # layout so we need to clear it before contuing.
        if (
            property_set["trivial_layout_score"] is not None
            and property_set["trivial_layout_score"] != 0
        ):
            return True
        return False

    def _vf2_match_not_found(property_set):
        # If a layout hasn't been set by the time we run vf2 layout we need to
        # run layout
        if property_set["layout"] is None:
            return True
        # if VF2 layout stopped for any reason other than solution found we need
        # to run layout since VF2 didn't converge.
        if (
            property_set["VF2Layout_stop_reason"] is not None
            and property_set["VF2Layout_stop_reason"] is not VF2LayoutStopReason.SOLUTION_FOUND
        ):
            return True
        return False

    _choose_layout_0 = (
        []
        if pass_manager_config.layout_method
        else [
            TrivialLayout(coupling_map),
            Layout2qDistance(coupling_map, property_name="trivial_layout_score"),
        ]
    )

    _choose_layout_1 = (
        []
        if pass_manager_config.layout_method
        else VF2Layout(
            coupling_map,
            seed=seed_transpiler,
            call_limit=int(5e4),  # Set call limit to ~100ms with retworkx 0.10.2
            time_limit=0.1,
            properties=backend_properties,
            target=target,
        )
    )

    # 2. Decompose so only 1-qubit and 2-qubit gates remain
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

    # 3. Use a better layout on densely connected qubits, if circuit needs swaps
    if layout_method == "trivial":
        _improve_layout = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _improve_layout = DenseLayout(coupling_map, backend_properties, target=target)
    elif layout_method == "noise_adaptive":
        _improve_layout = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _improve_layout = SabreLayout(coupling_map, max_iterations=2, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    # 4. Extend dag/layout with ancillas using the full coupling map
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 5. Swap to fit the coupling map
    _swap_check = CheckMap(coupling_map)

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    _swap = [BarrierBeforeFinalMeasurements()]
    if routing_method == "basic":
        _swap += [BasicSwap(coupling_map)]
    elif routing_method == "stochastic":
        _swap += [StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)]
    elif routing_method == "lookahead":
        _swap += [LookaheadSwap(coupling_map, search_depth=4, search_width=4)]
    elif routing_method == "sabre":
        _swap += [SabreSwap(coupling_map, heuristic="lookahead", seed=seed_transpiler)]
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

    # 6. Unroll to the basis
    if translation_method == "unroller":
        _unroll = [Unroller(basis_gates)]
    elif translation_method == "translator":
        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        _unroll = [
            # Use unitary synthesis for basis aware decomposition of UnitaryGates before
            # custom unrolling
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                method=unitary_synthesis_method,
                backend_props=backend_properties,
                plugin_config=unitary_synthesis_plugin_config,
                target=target,
            ),
            UnrollCustomDefinitions(sel, basis_gates),
            BasisTranslator(sel, basis_gates, target),
        ]
    elif translation_method == "synthesis":
        _unroll = [
            # Use unitary synthesis for basis aware decomposition of UnitaryGates before
            # collection
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                method=unitary_synthesis_method,
                backend_props=backend_properties,
                min_qubits=3,
                target=target,
            ),
            Unroll3qOrMore(),
            Collect2qBlocks(),
            ConsolidateBlocks(basis_gates=basis_gates, target=target),
            UnitarySynthesis(
                basis_gates,
                approximation_degree=approximation_degree,
                coupling_map=coupling_map,
                method=unitary_synthesis_method,
                backend_props=backend_properties,
                plugin_config=unitary_synthesis_plugin_config,
                target=target,
            ),
        ]
    else:
        raise TranspilerError("Invalid translation method %s." % translation_method)

    # 7. Fix any bad CX directions
    _direction_check = [CheckGateDirection(coupling_map, target)]

    def _direction_condition(property_set):
        return not property_set["is_direction_mapped"]

    _direction = [GateDirection(coupling_map, target)]

    # 8. Remove zero-state reset
    _reset = RemoveResetInZeroState()

    # 9. Merge 1q rotations and cancel CNOT gates iteratively until no more change in depth
    # or size of circuit
    _depth_check = [Depth(), FixedPoint("depth")]
    _size_check = [Size(), FixedPoint("size")]

    def _opt_control(property_set):
        return (not property_set["depth_fixed_point"]) or (not property_set["size_fixed_point"])

    _opt = [Optimize1qGatesDecomposition(basis_gates), CXCancellation()]

    # Build pass manager
    pm1 = PassManager()
    if coupling_map or initial_layout:
        pm1.append(_given_layout)
        pm1.append(_unroll3q)
        pm1.append(_choose_layout_0, condition=_choose_layout_condition)
        pm1.append(_choose_layout_1, condition=_trivial_not_perfect)
        pm1.append(_improve_layout, condition=_vf2_match_not_found)
        pm1.append(_embed)
        pm1.append(_swap_check)
        pm1.append(_swap, condition=_swap_condition)
    pm1.append(_unroll)
    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pm1.append(_direction_check)
        pm1.append(_direction, condition=_direction_condition)
    pm1.append(_reset)
    pm1.append(_depth_check + _size_check)
    pm1.append(_opt + _unroll + _depth_check + _size_check, do_while=_opt_control)
    if inst_map and inst_map.has_custom_gate():
        pm1.append(PulseGates(inst_map=inst_map))

    # 10. Unify all durations (either SI, or convert to dt if known)
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
        pm1.append(TimeUnitConversion(instruction_durations))
        try:
            pm1.append(scheduler[scheduling_method](instruction_durations))
        except KeyError as ex:
            raise TranspilerError("Invalid scheduling method %s." % scheduling_method) from ex
    elif instruction_durations:
        # No scheduling. But do unit conversion for delays.
        def _contains_delay(property_set):
            return property_set["contains_delay"]

        pm1.append(ContainsInstruction("delay"))
        pm1.append(TimeUnitConversion(instruction_durations), condition=_contains_delay)
    if (
        timing_constraints.granularity != 1
        or timing_constraints.min_length != 1
        or timing_constraints.acquire_alignment != 1
        or timing_constraints.pulse_alignment != 1
    ):
        # Run alignment analysis regardless of scheduling.

        def _require_alignment(property_set):
            return property_set["reschedule_required"]

        pm1.append(
            InstructionDurationCheck(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            )
        )
        pm1.append(
            ConstrainedReschedule(
                acquire_alignment=timing_constraints.acquire_alignment,
                pulse_alignment=timing_constraints.pulse_alignment,
            ),
            condition=_require_alignment,
        )
        pm1.append(
            ValidatePulseGates(
                granularity=timing_constraints.granularity,
                min_length=timing_constraints.min_length,
            )
        )
    if scheduling_method:
        # Call padding pass if circuit is scheduled
        pm1.append(PadDelay())

    return pm1
