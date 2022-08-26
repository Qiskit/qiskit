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
from qiskit.transpiler.passmanager import StagedPassManager

from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import TranspilerError
from qiskit.utils.optionals import HAS_TOQM


def level_0_pass_manager(pass_manager_config: PassManagerConfig) -> StagedPassManager:
    """Level 0 pass manager: no explicit optimization other than mapping to backend.

    This pass manager applies the user-given initial layout. If none is given, a trivial
    layout consisting of mapping the i-th virtual qubit to the i-th physical qubit is used.
    Any unused physical qubit is allocated as ancilla space.

    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map.

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

    # Choose an initial layout if not set by user (default: trivial layout)
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

    toqm_pass = False
    # Choose routing pass
    if routing_method == "basic":
        routing_pass = BasicSwap(coupling_map)
    elif routing_method == "stochastic":
        routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
    elif routing_method == "lookahead":
        routing_pass = LookaheadSwap(coupling_map, search_depth=2, search_width=2)
    elif routing_method == "sabre":
        routing_pass = SabreSwap(coupling_map, heuristic="basic", seed=seed_transpiler)
    elif routing_method == "toqm":
        HAS_TOQM.require_now("TOQM-based routing")
        from qiskit_toqm import ToqmSwap, ToqmStrategyO0, latencies_from_target

        if initial_layout:
            raise TranspilerError("Initial layouts are not supported with TOQM-based routing.")

        toqm_pass = True
        # Note: BarrierBeforeFinalMeasurements is skipped intentionally since ToqmSwap
        #       does not yet support barriers.
        routing_pass = ToqmSwap(
            coupling_map,
            strategy=ToqmStrategyO0(
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

    unroll_3q = None
    # Build pass manager
    if coupling_map or initial_layout:
        unroll_3q = common.generate_unroll_3q(
            target,
            basis_gates,
            approximation_degree,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
        )
        layout = PassManager()
        layout.append(_given_layout)
        layout.append(_choose_layout, condition=_choose_layout_condition)
        layout += common.generate_embed_passmanager(coupling_map)
        routing = common.generate_routing_passmanager(
            routing_pass,
            target,
            coupling_map=coupling_map,
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

    if (coupling_map and not coupling_map.is_symmetric) or (
        target is not None and target.get_non_global_operation_names(strict_direction=True)
    ):
        pre_opt = common.generate_pre_op_passmanager(target, coupling_map)
        pre_opt += translation
    else:
        pre_opt = None
    sched = common.generate_scheduling(
        instruction_durations, scheduling_method, timing_constraints, inst_map
    )

    return StagedPassManager(
        init=unroll_3q,
        layout=layout,
        pre_routing=pre_routing,
        routing=routing,
        translation=translation,
        pre_optimization=pre_opt,
        scheduling=sched,
    )
