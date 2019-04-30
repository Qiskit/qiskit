# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Circuit transpile function"""

from qiskit.transpiler.preset_passmanagers import (default_pass_manager_simulator,
                                                   default_pass_manager)


def transpile_circuit(circuit, transpile_config):
    """Select a PassManager and run a single circuit through it.

    Args:
        circuit (QuantumCircuit): circuit to transpile
        transpile_config (TranspileConfig): configuration dictating how to transpile

    Returns:
        QuantumCircuit: transpiled circuit
    """

    # if the pass manager is not already selected, choose an appropriate one.
    if transpile_config.pass_manager:
        pass_manager = transpile_config.pass_manager

    elif transpile_config.coupling_map:
        pass_manager = default_pass_manager(transpile_config.basis_gates,
                                            transpile_config.coupling_map,
                                            transpile_config.initial_layout,
                                            transpile_config.seed_transpiler)
    else:
        pass_manager = default_pass_manager_simulator(transpile_config.basis_gates)

    return pass_manager.run(circuit)
