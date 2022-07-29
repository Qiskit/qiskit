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

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


def _get_vf2_call_limit(pass_manager_config):
    """Get the vf2 call limit for vf2 based layout passes."""
    vf2_call_limit = None
    opt_level = pass_manager_config.optimization_level
    if pass_manager_config.layout_method is None and pass_manager_config.initial_layout is None:
        if opt_level == 1:
            vf2_call_limit = int(5e4)  # Set call limit to ~100ms with retworkx 0.10.2
        elif opt_level == 2:
            vf2_call_limit = int(5e6)  # Set call limit to ~10 sec with retworkx 0.10.2
        elif opt_level == 3:
            vf2_call_limit = int(3e7)  # Set call limit to ~60 sec with retworkx 0.10.2
    return vf2_call_limit


class BasicSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.BasicSwap`"""

    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
        """Build routing stage PassManager."""
        opt_level = pass_manager_config.optimization_level
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        routing_pass = BasicSwap(coupling_map)
        vf2_call_limit = _get_vf2_call_limit(pass_manager_config)
        if opt_level == 0:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 1:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 2:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 3:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        else:
            raise TranspilerError(f"Invalid optimization level specified: {opt_level}")


class StochasticSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.StochasticSwap`"""

    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
        """Build routing stage PassManager."""
        opt_level = pass_manager_config.optimization_level
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit = _get_vf2_call_limit(pass_manager_config)
        if opt_level == 0:
            routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 1:
            routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 2:
            routing_pass = StochasticSwap(coupling_map, trials=20, seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 3:
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        else:
            raise TranspilerError(f"Invalid optimization level specified: {opt_level}")


class LookaheadSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.LookaheadSwap`"""

    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
        """Build routing stage PassManager."""
        opt_level = pass_manager_config.optimization_level
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit = _get_vf2_call_limit(pass_manager_config)
        if opt_level == 0:
            routing_pass = LookaheadSwap(coupling_map, search_depth=2, search_width=2)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 1:
            routing_pass = LookaheadSwap(coupling_map, search_depth=4, search_width=4)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 2:
            routing_pass = LookaheadSwap(coupling_map, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 3:
            routing_pass = LookaheadSwap(coupling_map, search_depth=5, search_width=6)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        else:
            raise TranspilerError(f"Invalid optimization level specified: {opt_level}")


class SabreSwapPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with :class:`~.SabreSwap`"""

    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
        """Build routing stage PassManager."""
        opt_level = pass_manager_config.optimization_level
        seed_transpiler = pass_manager_config.seed_transpiler
        target = pass_manager_config.target
        coupling_map = pass_manager_config.coupling_map
        backend_properties = pass_manager_config.backend_properties
        vf2_call_limit = _get_vf2_call_limit(pass_manager_config)
        if opt_level == 0:
            routing_pass = SabreSwap(coupling_map, heuristic="basic", seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 1:
            routing_pass = SabreSwap(coupling_map, heuristic="lookahead", seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                check_trivial=True,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 2:
            routing_pass = SabreSwap(coupling_map, heuristic="decay", seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        elif opt_level == 3:
            routing_pass = SabreSwap(coupling_map, heuristic="decay", seed=seed_transpiler)
            return common.generate_routing_passmanager(
                routing_pass,
                target,
                coupling_map=coupling_map,
                vf2_call_limit=vf2_call_limit,
                backend_properties=backend_properties,
                seed_transpiler=seed_transpiler,
                use_barrier_before_measurement=True,
            )
        else:
            raise TranspilerError(f"Invalid optimization level specified: {opt_level}")


class NoneRoutingPassManager(PassManagerStagePlugin):
    """Plugin class for routing stage with error on routing."""

    def pass_manager(self, pass_manager_config: PassManagerConfig) -> PassManager:
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
