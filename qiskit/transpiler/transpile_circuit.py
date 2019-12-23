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
from qiskit.transpiler.passes.ms_basis_decomposer import MSBasisDecomposer
from qiskit.transpiler.exceptions import TranspilerError


def transpile_circuit(circuit, transpile_config):
    """Select a PassManager and run a single circuit through it.
    Args:
        circuit (QuantumCircuit): circuit to transpile
        transpile_config (dict): configuration dictating how to transpile.
            The format of the dictionary has the following format:
            {'circuit': QuantumCircuit,
             'optimization_level': int,
             'pass_manager': PassManager,
             'output_name': string,
             'callback': callable,
             'pass_manager_config': PassManagerConfig}
    Returns:
        QuantumCircuit: transpiled circuit
    Raises:
        TranspilerError: if transpile_config is not valid or transpilation incurs error
    """
    # either the pass manager is already selected...
    if transpile_config['pass_manager'] is not None:
        pass_manager = transpile_config['pass_manager']
    # or we choose an appropriate one based on desired optimization level (default: level 1)
    else:
        level = transpile_config['optimization_level']

        # Workaround for ion trap support: If basis gates includes
        # Mølmer-Sørensen (rxx) and the circuit includes gates outside the basis,
        # first unroll to u3, cx, then run MSBasisDecomposer to target basis.
        basic_insts = ['measure', 'reset', 'barrier', 'snapshot']
        device_insts = set(transpile_config['pass_manager_config'].basis_gates).union(basic_insts)

        ms_basis_swap = None
        if 'rxx' in transpile_config['pass_manager_config'].basis_gates and \
                not device_insts >= transpile_config['circuit'].count_ops().keys():
            ms_basis_swap = transpile_config['pass_manager_config'].basis_gates
            transpile_config['pass_manager_config'].basis_gates = list(set(['u3', 'cx']).union(
                transpile_config['pass_manager_config'].basis_gates))

        if level is None:
            level = 1

        if level == 0:
            pass_manager = level_0_pass_manager(transpile_config['pass_manager_config'])
        elif level == 1:
            pass_manager = level_1_pass_manager(transpile_config['pass_manager_config'])
        elif level == 2:
            pass_manager = level_2_pass_manager(transpile_config['pass_manager_config'])
        elif level == 3:
            pass_manager = level_3_pass_manager(transpile_config['pass_manager_config'])
        else:
            raise TranspilerError("optimization_level can range from 0 to 3.")

        if ms_basis_swap is not None:
            pass_manager.append(MSBasisDecomposer(ms_basis_swap))

    out_circuit = pass_manager.run(circuit,
                                   callback=transpile_config['callback'],
                                   output_name=transpile_config['output_name'])

    return out_circuit
