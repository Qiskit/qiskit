# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Built-in transpiler stage plugins for preset pass managers."""

from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.preset_passmanagers.plugin import (
    PassManagerStagePluginManager,
)
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    Collect2qBlocks,
    ConsolidateBlocks,
    CXCancellation,
)
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController


class BasicSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.BasicSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        if target is None:
            routing_pass = BasicSwap(coupling_map)
        else:
            routing_pass = BasicSwap(target)

        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class StochasticSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.StochasticSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 3:
            routing_pass = StochasticSwap(coupling_map_routing, trials=200, seed=seed_transpiler)
        else:
            routing_pass = StochasticSwap(coupling_map_routing, trials=20, seed=seed_transpiler)

        if optimization_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level in {2, 3}:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class LookaheadSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.LookaheadSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=2, search_width=2)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=4, search_width=4)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            routing_pass = LookaheadSwap(coupling_map_routing, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class SabreSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.SabreSwap`"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        coupling_map_routing = target
        if coupling_map_routing is None:
            coupling_map_routing = coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit, vf2_max_trials = common.get_vf2_limits(
            optimization_level,
            pass_manager_config.layout_method,
            pass_manager_config.initial_layout,
        )
        if optimization_level == 0:
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="basic",
                seed=seed_transpiler,
                trials=5,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 1:
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=5,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 2:
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=10,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        if optimization_level == 3:
            routing_pass = SabreSwap(
                coupling_map_routing,
                heuristic="decay",
                seed=seed_transpiler,
                trials=20,
            )
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                vf2_max_trials=vf2_max_trials,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        raise TranspilerError(f"Invalid optimization level specified: {optimization_level}")


class NoneRoutingPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with error on routing."""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build routing stage PassManager."""
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        routing_pass = Error(
            msg="No routing method selected, but circuit is not routed to device. "
            "CheckMap Error: {check_map_msg}",
            action="raise",
        )
        return common.generate_routing_passmanager(
            routing_pass,
            target,
            coupling_map=coupling_map,
            seed_transpiler=seed_transpiler,
            use_barrier_before_measurement=True,
        )


class OptimizationPassManager(PassManagerStagePlugin):
    """Plugin class for optimization stage"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        """Build pass manager for optimization stage."""
        # Obtain the translation method required for this pass to work
        translation_method = pass_manager_config.translation_method or "translator"
        optimization = PassManager()
        if optimization_level != 0:
            plugin_manager = PassManagerStagePluginManager()
            _depth_check = [Depth(recurse=True), FixedPoint("depth")]
            _size_check = [Size(recurse=True), FixedPoint("size")]
            # Minimum point check for optimization level 3.
            _minimum_point_check = [
                Depth(recurse=True),
                Size(recurse=True),
                MinimumPoint(["depth", "size"], "optimization_loop"),
            ]

            def _opt_control(property_set):
                return (not property_set["depth_fixed_point"]) or (
                    not property_set["size_fixed_point"]
                )

            if translation_method not in {
                "translator",
                "synthesis",
                "unroller",
            }:
                translation = plugin_manager.get_passmanager_stage(
                    "translation",
                    translation_method,
                    pass_manager_config,
                    optimization_level=1,
                )
            else:
                translation = common.generate_translation_passmanager(
                    pass_manager_config.target,
                    pass_manager_config.basis_gates,
                    translation_method,
                    pass_manager_config.approximation_degree,
                    pass_manager_config.coupling_map,
                    pass_manager_config.backend_properties,
                    pass_manager_config.unitary_synthesis_method,
                    pass_manager_config.unitary_synthesis_plugin_config,
                    pass_manager_config.hls_config,
                )

            if optimization_level == 2:
                # Steps for optimization level 2
                _opt = [
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(
                        basis_gates=pass_manager_config.basis_gates,
                        target=pass_manager_config.target,
                    ),
                ]

            elif optimization_level == 3:
                # Steps for optimization level 3
                _opt = [
                    Collect2qBlocks(),
                    ConsolidateBlocks(
                        basis_gates=pass_manager_config.basis_gates,
                        target=pass_manager_config.target,
                        approximation_degree=pass_manager_config.approximation_degree,
                    ),
                    UnitarySynthesis(
                        pass_manager_config.basis_gates,
                        approximation_degree=pass_manager_config.approximation_degree,
                        coupling_map=pass_manager_config.coupling_map,
                        backend_props=pass_manager_config.backend_properties,
                        method=pass_manager_config.unitary_synthesis_method,
                        plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                        target=pass_manager_config.target,
                    ),
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CommutativeCancellation(target=pass_manager_config.target),
                ]

                def _opt_control(property_set):
                    return not property_set["optimization_loop_minimum_point"]

            else:
                # Steps for optimization level 1
                _opt = [
                    Optimize1qGatesDecomposition(
                        basis=pass_manager_config.basis_gates, target=pass_manager_config.target
                    ),
                    CXCancellation(),
                ]

            unroll = [pass_ for x in translation.passes() for pass_ in x["passes"]]
            # Build nested Flow controllers
            def _unroll_condition(property_set):
                return not property_set["all_gates_in_basis"]

            # Check if any gate is not in the basis, and if so, run unroll passes
            _unroll_if_out_of_basis = [
                GatesInBasis(pass_manager_config.basis_gates, target=pass_manager_config.target),
                ConditionalController(unroll, condition=_unroll_condition),
            ]

            if optimization_level == 3:
                optimization.append(_minimum_point_check)
            else:
                optimization.append(_depth_check + _size_check)
            opt_loop = (
                _opt + _unroll_if_out_of_basis + _minimum_point_check
                if optimization_level == 3
                else _opt + _unroll_if_out_of_basis + _depth_check + _size_check
            )
            optimization.append(opt_loop, do_while=_opt_control)
            return optimization
        else:
            return None
