# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Preset pass manager generation function for compiling into Clifford+T.
"""

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.builtin_plugins import (
    CliffordTInitPassManager,
    DefaultLayoutPassManager,
    DefaultRoutingPassManager,
    DefaultSchedulingPassManager,
    OptimizeCliffordRZPassManager,
    OptimizeCliffordTPassManager,
    TranslateToCliffordRZPassManager,
    TranslateToCliffordTPassManager,
)


def generate_clifford_t_pass_manager(
    pass_manager_config: PassManagerConfig, optimization_level: int
) -> StagedPassManager:
    """Generate Clifford+T stage pass manager.

    Args:
        pass_manager_config: configuration of the pass manager.
        optimization_level: The optimization level. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time.
    Returns:
        Stage pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise TranspilerError(f"Invalid optimization level {optimization_level}")

    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    if coupling_map or initial_layout:
        layout = DefaultLayoutPassManager().pass_manager(
            pass_manager_config, optimization_level=optimization_level
        )
        routing = DefaultRoutingPassManager().pass_manager(
            pass_manager_config, optimization_level=optimization_level
        )
    else:
        layout = None
        routing = None

    init = CliffordTInitPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    rz_translation = TranslateToCliffordRZPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    rz_optimization = OptimizeCliffordRZPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    t_translation = TranslateToCliffordTPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    t_optimization = OptimizeCliffordTPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    sched = DefaultSchedulingPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )

    stages = [
        "init",
        "layout",
        "routing",
        "rz_translation",
        "rz_optimization",
        "t_translation",
        "t_optimization",
        "scheduling",
    ]

    return StagedPassManager(
        stages=stages,
        init=init,
        layout=layout,
        routing=routing,
        rz_translation=rz_translation,
        rz_optimization=rz_optimization,
        t_translation=t_translation,
        t_optimization=t_optimization,
        scheduling=sched,
    )
