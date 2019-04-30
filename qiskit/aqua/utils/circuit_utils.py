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
import numpy as np
from qiskit import compiler, BasicAer, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller


def convert_to_basis_gates(circuit):
    # unroll the circuit using the basis u1, u2, u3, cx, and id gates
    unroller = Unroller(basis=['u1', 'u2', 'u3', 'cx', 'id'])
    pm = PassManager(passes=[unroller])
    qc = compiler.transpile(circuit, BasicAer.get_backend('qasm_simulator'), pass_manager=pm)
    return qc


def is_qubit(qb):
    # check if the input is a qubit, which is in the form (QuantumRegister, int)
    return isinstance(qb, tuple) and isinstance(qb[0], QuantumRegister) and isinstance(qb[1], int)


def is_qubit_list(qbs):
    # check if the input is a list of qubits
    for qb in qbs:
        if not is_qubit(qb):
            return False
    return True


def summarize_circuits(circuits):
    """Summarize circuits based on QuantumCircuit, and four metrics are summarized.

    Number of qubits and classical bits, and number of operations and depth of circuits.
    The average statistic is provided if multiple circuits are inputed.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    ret += "Submitting {} circuits.\n".format(len(circuits))
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, circuit in enumerate(circuits):
        dag = circuit_to_dag(circuit)
        depth = dag.depth()
        width = dag.width()
        size = dag.size()
        classical_bits = dag.num_cbits()
        op_counts = dag.count_ops()
        stats[0] += width
        stats[1] += classical_bits
        stats[2] += size
        stats[3] += depth
        ret = ''.join([ret, "{}-th circuit: {} qubits, {} classical bits and {} operations with depth {}\n op_counts: {}\n".format(
            i, width, classical_bits, size, depth, op_counts)])
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = ''.join([ret, "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} operations with depth {:.2f}\n".format(
            stats[0], stats[1], stats[2], stats[3])])
    ret += "============================================================================\n"
    return ret
