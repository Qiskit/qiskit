# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Circuit utility functions """

import numpy as np
from qiskit import compiler
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller


def convert_to_basis_gates(circuit):
    """ unroll the circuit using the basis u1, u2, u3, cx, and id gates """
    # pylint: disable=import-outside-toplevel
    from qiskit import BasicAer
    unroller = Unroller(basis=['u1', 'u2', 'u3', 'cx', 'id'])
    p_m = PassManager(passes=[unroller])
    qc = compiler.transpile(circuit, BasicAer.get_backend('qasm_simulator'), pass_manager=p_m)
    return qc


def summarize_circuits(circuits):
    """Summarize circuits based on QuantumCircuit, and five metrics are summarized.
        - Number of qubits
        - Number of classical bits
        - Number of operations
        - Depth of circuits
        - Counts of different gate operations

    The average statistic of the first four is provided if multiple circuits are provided.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    Returns:
        str: a formatted string records the summary
    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    ret += "Submitting {} circuits.\n".format(len(circuits))
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, circuit in enumerate(circuits):
        depth = circuit.depth()
        size = circuit.size()
        num_qubits = sum(reg.size for reg in circuit.qregs)
        num_clbits = sum(reg.size for reg in circuit.cregs)
        op_counts = circuit.count_ops()
        stats[0] += num_qubits
        stats[1] += num_clbits
        stats[2] += size
        stats[3] += depth
        ret = ''.join([
            ret,
            "{}-th circuit: {} qubits, {} classical bits and {} "
            "operations with depth {}\nop_counts: {}\n".format(
                i, num_qubits, num_clbits, size, depth, op_counts
            )
        ])
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = ''.join([
            ret,
            "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} "
            "operations with depth {:.2f}\n".format(
                stats[0], stats[1], stats[2], stats[3]
            )
        ])
    ret += "============================================================================\n"
    return ret
