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
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StagedPassManager

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import FixedPoint
from qiskit.transpiler.passes import Depth
from qiskit.transpiler.passes import Size
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

from qiskit.transpiler import TranspilerError
from qiskit.utils.optionals import HAS_TOQM


def level_3_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
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

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 3 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    basis_gates = pass_manager_config.basis_gates
    inst_map = pass_manager_config.inst_map
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "sabre"
    routing_method = pass_manager_config.routing_method or "sabre"
    translation_method = pass_manager_config.translation_method or "translator"
    scheduling_method = pass_manager_config.scheduling_method
    instruction_durations = pass_manager_config.instruction_durations
    seed_transpiler = pass_manager_config.seed_transpiler
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    timing_constraints = pass_manager_config.timing_constraints or TimingConstraints()
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target

    # Layout on good qubits if calibration info available, otherwise on dense links
    _given_layout = SetLayout(initial_layout)

    def _choose_layout_condition(property_set):
        # layout hasn't been set yet
        return not property_set["layout"]

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

    # 2a. If layout method is not set, first try VF2Layout
    _choose_layout_0 = (
        []
        if pass_manager_config.layout_method
        else VF2Layout(
            coupling_map,
            seed=seed_transpiler,
            call_limit=int(3e7),  # Set call limit to ~60 sec with retworkx 0.10.2
            properties=backend_properties,
            target=target,
        )
    )
    # 2b. if VF2 didn't converge on a solution use layout_method (dense).
    if layout_method == "trivial":
        _choose_layout_1 = TrivialLayout(coupling_map)
    elif layout_method == "dense":
        _choose_layout_1 = DenseLayout(coupling_map, backend_properties, target=target)
    elif layout_method == "noise_adaptive":
        _choose_layout_1 = NoiseAdaptiveLayout(backend_properties)
    elif layout_method == "sabre":
        _choose_layout_1 = SabreLayout(coupling_map, max_iterations=4, seed=seed_transpiler)
    else:
        raise TranspilerError("Invalid layout method %s." % layout_method)

    toqm_pass = False
    if routing_method == "basic":
        routing_pass = BasicSwap(coupling_map)
    elif routing_method == "stochastic":
        routing_pass = StochasticSwap(coupling_map, trials=200, seed=seed_transpiler)
    elif routing_method == "lookahead":
        routing_pass = LookaheadSwap(coupling_map, search_depth=5, search_width=6)
    elif routing_method == "sabre":
        routing_pass = SabreSwap(coupling_map, heuristic="decay", seed=seed_transpiler)
    elif routing_method == "toqm":
        HAS_TOQM.require_now("TOQM-based routing")
        from qiskit_toqm import ToqmSwap, ToqmStrategyO3, latencies_from_target

        if initial_layout:
            raise TranspilerError("Initial layouts are not supported with TOQM-based routing.")

        toqm_pass = True
        # Note: BarrierBeforeFinalMeasurements is skipped intentionally since ToqmSwap
        #       does not yet support barriers.
        routing_pass = ToqmSwap(
            coupling_map,
            strategy=ToqmStrategyO3(
                latencies_from_target(
                    coupling_map, instruction_durations, basis_gates, backend_properties, target
                )
            ),
        )
    elif routing_method == "none":
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
    else:
        raise TranspilerError("Invalid routing method %s." % routing_method)

    # 8. Optimize iteratively until no more change in depth. Removes useless gates
    # after reset and before measure, commutes gates and optimizes contiguous blocks.
    _depth_check = [Depth(), FixedPoint("depth")]
    _size_check = [Size(), FixedPoint("size")]

    def _opt_control(property_set):
        return (not property_set["depth_fixed_point"]) or (not property_set["size_fixed_point"])

    _opt = [
        Collect2qBlocks(),
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
        Optimize1qGatesDecomposition(basis_gates),
        CommutativeCancellation(),
    ]

    # Build pass manager
    init = common.generate_unroll_3q(
        target,
        basis_gates,
        approximation_degree,
        unitary_synthesis_method,
        unitary_synthesis_plugin_config,
    )
    init.append(RemoveResetInZeroState())
    init.append(OptimizeSwapBeforeMeasure())
    init.append(RemoveDiagonalGatesBeforeMeasure())
    if coupling_map or initial_layout:
        layout = PassManager()
        layout.append(_given_layout)
        layout.append(_choose_layout_0, condition=_choose_layout_condition)
        layout.append(_choose_layout_1, condition=_vf2_match_not_found)
        layout += common.generate_embed_passmanager(coupling_map)
        vf2_call_limit = None
        if pass_manager_config.layout_method is None and pass_manager_config.initial_layout is None:
            vf2_call_limit = int(3e7)  # Set call limit to ~60 sec with retworkx 0.10.2
        routing = common.generate_routing_passmanager(
            routing_pass,
            target,
            coupling_map=coupling_map,
            vf2_call_limit=vf2_call_limit,
            backend_properties=backend_properties,
            seed_transpiler=seed_transpiler,
            use_barrier_before_measurement=not toqm_pass,
        )
    else:
        layout = None
        routing = None
    translation = common.generate_translation_passmanager(
        target,
        basis_gates,
        translation_method,
        approximation_degree,
        coupling_map,
        backend_properties,
        unitary_synthesis_method,
        unitary_synthesis_plugin_config,
    )
    pre_routing = None
    if toqm_pass:
        pre_routing = translation
    optimization = PassManager()
    unroll = [pass_ for x in translation.passes() for pass_ in x["passes"]]
    optimization.append(_depth_check + _size_check)
    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pre_optimization = common.generate_pre_op_passmanager(target, coupling_map, True)
        _direction = [
            pass_
            for x in common.generate_pre_op_passmanager(target, coupling_map).passes()
            for pass_ in x["passes"]
        ]
        # For transpiling to a target we need to run GateDirection in the
        # optimization loop to correct for incorrect directions that might be
        # inserted by UnitarySynthesis which is direction aware but only via
        # the coupling map which with a target doesn't give a full picture
        if target is not None:
            optimization.append(
                _opt + unroll + _depth_check + _size_check + _direction, do_while=_opt_control
            )
        else:
            optimization.append(_opt + unroll + _depth_check + _size_check, do_while=_opt_control)
    else:
        pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=True)
        optimization.append(_opt + unroll + _depth_check + _size_check, do_while=_opt_control)
    opt_loop = _depth_check + _opt + unroll
    optimization.append(opt_loop, do_while=_opt_control)
    sched = common.generate_scheduling(
        instruction_durations, scheduling_method, timing_constraints, inst_map
    )
    return StagedPassManager(
        init=init,
        layout=layout,
        pre_routing=pre_routing,
        routing=routing,
        translation=translation,
        pre_optimization=pre_optimization,
        optimization=optimization,
        scheduling=sched,
    )
