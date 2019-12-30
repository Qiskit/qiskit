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


def transpile_circuit(circuit,
                      pass_manager,
                      output_name: None,
                      callback: None):
    """Select a PassManager and run a single circuit through it.
    Args:
        circuit (QuantumCircuit): circuit to transpile
        pass_manager (PassManager): The pass manager to use for a custom pipeline of
            transpiler passes.
        output_name (string): To identify the output circuits
        callback (callable): Function that will be called after each pass execution.

    Returns:
        QuantumCircuit: transpiled circuit
    """
    out_circuit = pass_manager.run(circuit, callback=callback, output_name=output_name)

    return out_circuit
