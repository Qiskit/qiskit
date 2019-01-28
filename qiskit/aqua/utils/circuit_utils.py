# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np


def cry(theta, q_control, q_target, qc):
    qc.u3(theta / 2, 0, 0, q_target)
    qc.cx(q_control, q_target)
    qc.u3(-theta / 2, 0, 0, q_target)
    qc.cx(q_control, q_target)
    return qc


def ccry(theta, q_control_1, q_control_2, q_target, qc):
    qc.u3(theta / 2, 0, 0, q_target)
    qc.ccx(q_control_1, q_control_2, q_target)
    qc.u3(-theta / 2, 0, 0, q_target)
    qc.ccx(q_control_1, q_control_2, q_target)
    return qc


def multi_cry(theta, controls, target, ancillas, qc, q):
    qc.u3(theta / 2, 0, 0, q[target])
    multi_toffoli(qc, q, controls, target, ancillas)
    qc.u3(-theta / 2, 0, 0, q[target])
    multi_toffoli(qc, q, controls, target, ancillas)
    return qc


def multi_cry_q(theta, q_controls, q_target, q_ancillas, qc):
    qc.u3(theta / 2, 0, 0, q_target)
    multi_toffoli_q(qc, q_controls, q_target, q_ancillas)
    qc.u3(-theta / 2, 0, 0, q_target)
    multi_toffoli_q(qc, q_controls, q_target, q_ancillas)
    return qc


def controlled_hadamard(qc, q_control, q_target):
    qc.ry(-7 / 4 * np.pi, q_target)
    qc.cx(q_control, q_target)
    qc.ry(7 / 4 * np.pi, q_target)


def multi_toffoli(qc, q, controls, target, ancillas=None):
    """
    N = number of qubits
    controls = control qubits
    target = target qubit
    ancillas = ancilla qubits, len(ancillas) = len(controls) - 2
    """
    if len(controls) == 1:
        qc.cx(q[controls[0]], q[target])
    if len(controls) == 2:
        qc.ccx(q[controls[0]], q[controls[1]], q[target])
    elif len(controls) > 2 and (ancillas is None or len(ancillas) < len(controls) - 2):
        raise Exception('ERROR: need more ancillas for multi_toffoli!')
    else:
        multi_toffoli(qc, q, controls[:-1], ancillas[-1], ancillas[:-1])
        qc.ccx(q[controls[-1]], q[ancillas[-1]], q[target])
        multi_toffoli(qc, q, controls[:-1], ancillas[-1], ancillas[:-1])


def multi_toffoli_q(qc, q_controls, q_target, q_ancillas=None):
    """
    N = number of qubits
    controls = control qubits
    target = target qubit
    ancillas = ancilla qubits, len(ancillas) = len(controls) - 2
    """

    q_controls = register_to_list(q_controls)
    q_ancillas = register_to_list(q_ancillas)

    if len(q_controls) == 1:
        qc.cx(q_controls[0], q_target)
    elif len(q_controls) == 2:
        qc.ccx(q_controls[0], q_controls[1], q_target)
    elif len(q_controls) > 2 and (q_ancillas is None or len(q_ancillas) < len(q_controls) - 2):
        raise Exception('ERROR: need more ancillas for multi_toffoli!')
    else:
        multi_toffoli_q(qc, q_controls[:-1], q_ancillas[-1], q_ancillas[:-1])
        qc.ccx(q_controls[-1], q_ancillas[-1], q_target)
        multi_toffoli_q(qc, q_controls[:-1], q_ancillas[-1], q_ancillas[:-1])


def reverse_qubits(qc, q, targets):
    for i in range(int(np.floor(len(targets) / 2))):
        qc.swap(q[targets[i]], q[targets[-(1 + i)]])


def iqft(qc, q, i_targets, swaps=True):
    """n-qubit Inverse QFT on q in cq."""
    n = len(i_targets)

    if swaps:
        reverse_qubits(qc, q, i_targets)

    for j in range(n)[::-1]:
        qc.h(q[i_targets[j]])
        for k in range(j)[::-1]:
            theta_jk = -np.pi / float(2 ** (j - k))
            qc.cu1(theta_jk, q[i_targets[j]], q[i_targets[k]])


def logical_or(qc, q, i_a, i_b, i_c):
    qc.x(q[i_a])
    qc.x(q[i_b])
    qc.x(q[i_c])

    qc.ccx(q[i_a], q[i_b], q[i_c])

    qc.x(q[i_a])
    qc.x(q[i_b])


def logical_multi_or(qc, q_in, q_out, q_ancillas):
    for q in q_in:
        qc.x(q)
    multi_toffoli_q(qc, q_in, q_out, q_ancillas)
    for q in q_in:
        qc.x(q)
    qc.x(q_out)


def logical_or_inverse(qc, q, i_a, i_b, i_c):
    qc.x(q[i_b])
    qc.x(q[i_a])

    qc.ccx(q[i_a], q[i_b], q[i_c])

    qc.x(q[i_c])
    qc.x(q[i_b])
    qc.x(q[i_a])


def register_to_list(q, start=0, end=None):
    if q is None:
        return None
    if type(q) is tuple:
        return [q]
    if end is None:
        end = len(q)
    return [q[i] for i in range(start, end)]
