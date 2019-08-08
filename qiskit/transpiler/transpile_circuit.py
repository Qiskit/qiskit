# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit transpile function"""

from qiskit.transpiler.preset_passmanagers import (level_0_pass_manager,
                                                   level_1_pass_manager,
                                                   level_2_pass_manager,
                                                   level_3_pass_manager)
from qiskit.transpiler.exceptions import TranspilerError


def transpile_circuit(circuit, transpile_config):
    """Select a PassManager and run a single circuit through it.

    Args:
        circuit (QuantumCircuit): circuit to transpile
        transpile_config (TranspileConfig): configuration dictating how to transpile

    Returns:
        QuantumCircuit: transpiled circuit

    Raises:
        TranspilerError: if transpile_config is not valid or transpilation incurs error
    """
    # either the pass manager is already selected...
    if transpile_config.pass_manager:
        pass_manager = transpile_config.pass_manager

    # or we choose an appropriate one based on desired optimization level (default: level 1)
    else:
        level = transpile_config.optimization_level
        if level is None:
            level = 1

        if level == 0:
            pass_manager = level_0_pass_manager(transpile_config)
        elif level == 1:
            pass_manager = level_1_pass_manager(transpile_config)
        elif level == 2:
            pass_manager = level_2_pass_manager(transpile_config)
        elif level == 3:
            pass_manager = level_3_pass_manager(transpile_config)
        else:
            raise TranspilerError("optimization_level can range from 0 to 3.")

    # Set a callback on the pass manager there is one
    if getattr(transpile_config, 'callback', None):
        pass_manager.callback = transpile_config.callback

    out_circuit = pass_manager.run(circuit)
    out_circuit.name = transpile_config.output_name

    return out_circuit
