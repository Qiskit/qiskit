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
Preset pass manager generation function for compiling into PBC.
"""

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.passmanager_config import PassManagerPBCConfig
from qiskit.transpiler.preset_passmanagers.builtin_plugins import (
    PBCUnrollPassManager,
    PBCOptimizePassManager,
    PBCTranslateToPBCPassManager,
    PBCOptimizePBCPassManager,
)


def pbc_pass_manager(
    pass_manager_config: PassManagerPBCConfig, optimization_level: int
) -> StagedPassManager:
    """Generate a staged pass manager for transpiling into PBC.

    This function is invoked by :func:`.generate_preset_pbc_pass_manager`.
    It generates a specialized transpilation pipeline consisting of the following stages:

    * Unrolling: Decompose circuit instructions into a basis consisting of
      standard gates and instructions, pauli product rotations and measurements, and
      control-flow operations.
    * Optimization: Optimize unrolled circuits.
    * PBC translation: Translate unrolled circuits with into Pauli-based circuits.
    * PBC optimization: Optimize Pauli-based circuits.

    Args:
        pass_manager_config: Configuration of the pass manager.
        optimization_level: The optimization level. By default optimization level 2
            is used if this is not specified. This can be 0, 1, 2, or 3. Higher
            levels generate potentially more optimized circuits, at the expense
            of longer transpilation time.
    Returns:
        Staged pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise TranspilerError(f"Invalid optimization level {optimization_level}")

    unrolling = PBCUnrollPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    optimization = PBCOptimizePassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    pbc_translation = PBCTranslateToPBCPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )
    pbc_optimization = PBCOptimizePBCPassManager().pass_manager(
        pass_manager_config, optimization_level=optimization_level
    )

    stages = [
        "unrolling",
        "optimization",
        "pbc_translation",
        "pbc_optimization",
    ]

    return StagedPassManager(
        stages=stages,
        unrolling=unrolling,
        optimization=optimization,
        pbc_translation=pbc_translation,
        pbc_optimization=pbc_optimization,
    )
