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
from __future__ import annotations
from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.passes import MinimumPoint
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
from qiskit.transpiler.passes import GatesInBasis
from qiskit.transpiler.runningpassmanager import ConditionalController, FlowController
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)


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
    plugin_manager = PassManagerStagePluginManager()
    basis_gates = pass_manager_config.basis_gates
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    init_method = pass_manager_config.init_method
    layout_method = pass_manager_config.layout_method or "default"
    routing_method = pass_manager_config.routing_method or "sabre"
    translation_method = pass_manager_config.translation_method or "translator"
    optimization_method = pass_manager_config.optimization_method
    scheduling_method = pass_manager_config.scheduling_method or "default"
    backend_properties = pass_manager_config.backend_properties
    approximation_degree = pass_manager_config.approximation_degree
    unitary_synthesis_method = pass_manager_config.unitary_synthesis_method
    unitary_synthesis_plugin_config = pass_manager_config.unitary_synthesis_plugin_config
    target = pass_manager_config.target
    hls_config = pass_manager_config.hls_config

    # Choose routing pass
    routing_pm = plugin_manager.get_passmanager_stage(
        "routing", routing_method, pass_manager_config, optimization_level=3
    )

    # 8. Optimize iteratively until no more change in depth. Removes useless gates
    # after reset and before measure, commutes gates and optimizes contiguous blocks.
    _minimum_point_check: list[BasePass | FlowController] = [
        Depth(recurse=True),
        Size(recurse=True),
        MinimumPoint(["depth", "size"], "optimization_loop"),
    ]

    def _opt_control(property_set):
        return not property_set["optimization_loop_minimum_point"]

    _opt: list[BasePass | FlowController] = [
        Collect2qBlocks(),
        ConsolidateBlocks(
            basis_gates=basis_gates, target=target, approximation_degree=approximation_degree
        ),
        UnitarySynthesis(
            basis_gates,
            approximation_degree=approximation_degree,
            coupling_map=coupling_map,
            backend_props=backend_properties,
            method=unitary_synthesis_method,
            plugin_config=unitary_synthesis_plugin_config,
            target=target,
        ),
        Optimize1qGatesDecomposition(basis=basis_gates, target=target),
        CommutativeCancellation(target=target),
    ]

    # Build pass manager
    init = common.generate_control_flow_options_check(
        layout_method=layout_method,
        routing_method=routing_method,
        translation_method=translation_method,
        optimization_method=optimization_method,
        scheduling_method=scheduling_method,
        basis_gates=basis_gates,
        target=target,
    )
    if init_method is not None:
        init += plugin_manager.get_passmanager_stage(
            "init", init_method, pass_manager_config, optimization_level=2
        )
    else:
        init += common.generate_unroll_3q(
            target,
            basis_gates,
            approximation_degree,
            unitary_synthesis_method,
            unitary_synthesis_plugin_config,
            hls_config,
        )
    init.append(RemoveResetInZeroState())
    init.append(OptimizeSwapBeforeMeasure())
    init.append(RemoveDiagonalGatesBeforeMeasure())
    if coupling_map or initial_layout:
        layout = plugin_manager.get_passmanager_stage(
            "layout", layout_method, pass_manager_config, optimization_level=3
        )
        routing = routing_pm
    else:
        layout = None
        routing = None

    translation = plugin_manager.get_passmanager_stage(
        "translation", translation_method, pass_manager_config, optimization_level=3
    )

    if optimization_method is None:
        optimization = PassManager()
        unroll = [pass_ for x in translation.passes() for pass_ in x["passes"]]
        # Build nested Flow controllers
        def _unroll_condition(property_set):
            return not property_set["all_gates_in_basis"]

        # Check if any gate is not in the basis, and if so, run unroll passes
        _unroll_if_out_of_basis: list[BasePass | FlowController] = [
            GatesInBasis(basis_gates, target=target),
            ConditionalController(unroll, condition=_unroll_condition),
        ]

        optimization.append(_minimum_point_check)
        if (coupling_map and not coupling_map.is_symmetric) or (
            target is not None and target.get_non_global_operation_names(strict_direction=True)
        ):
            pre_optimization = common.generate_pre_op_passmanager(target, coupling_map, True)
            _direction = [
                pass_
                for x in common.generate_pre_op_passmanager(target, coupling_map).passes()
                for pass_ in x["passes"]
            ]
            if optimization is not None:
                optimization.append(
                    _opt + _unroll_if_out_of_basis + _minimum_point_check,
                    do_while=_opt_control,
                )
        else:
            pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=True)
            optimization.append(
                _opt + _unroll_if_out_of_basis + _minimum_point_check, do_while=_opt_control
            )
    else:
        optimization = plugin_manager.get_passmanager_stage(
            "optimization", optimization_method, pass_manager_config, optimization_level=3
        )
        if (coupling_map and not coupling_map.is_symmetric) or (
            target is not None and target.get_non_global_operation_names(strict_direction=True)
        ):
            pre_optimization = common.generate_pre_op_passmanager(target, coupling_map, True)
        else:
            pre_optimization = common.generate_pre_op_passmanager(remove_reset_in_zero=True)

    sched = plugin_manager.get_passmanager_stage(
        "scheduling", scheduling_method, pass_manager_config, optimization_level=3
    )

    return StagedPassManager(
        init=init,
        layout=layout,
        routing=routing,
        translation=translation,
        pre_optimization=pre_optimization,
        optimization=optimization,
        scheduling=sched,
    )
