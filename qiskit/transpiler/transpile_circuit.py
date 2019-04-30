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
