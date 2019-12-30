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

from qiskit.transpiler.passes.ms_basis_decomposer import MSBasisDecomposer


def transpile_circuit(circuit,
                      pass_manager,
                      output_name,
                      callback,
                      pass_manager_config):
    """Select a PassManager and run a single circuit through it.
    Args:
        circuit (QuantumCircuit): circuit to transpile
        pass_manager (PassManager): The pass manager to use for a custom pipeline of
            transpiler passes.
        output_name (string): To identify the output circuits
        callback (callable): Function that will be called after each pass execution.
        pass_manager_config (PassManagerConfig): Configuration instance.

    Returns:
        QuantumCircuit: transpiled circuit
    """
    # Workaround for ion trap support: If basis gates includes
    # Mølmer-Sørensen (rxx) and the circuit includes gates outside the basis,
    # first unroll to u3, cx, then run MSBasisDecomposer to target basis.
    basic_insts = ['measure', 'reset', 'barrier', 'snapshot']
    device_insts = set(pass_manager_config.basis_gates).union(basic_insts)

    ms_basis_swap = None
    if 'rxx' in pass_manager_config.basis_gates and \
            not device_insts >= circuit.count_ops().keys():
        ms_basis_swap = pass_manager_config.basis_gates
        pass_manager_config.basis_gates = list(
            set(['u3', 'cx']).union(pass_manager_config.basis_gates))

    if ms_basis_swap is not None:
        pass_manager.append(MSBasisDecomposer(ms_basis_swap))

    out_circuit = pass_manager.run(circuit, callback=callback, output_name=output_name)

    return out_circuit
