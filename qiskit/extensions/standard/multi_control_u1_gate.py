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
"""
Multiple-Control U1 gate. Not using ancillary qubits.
"""

import logging
from numpy import angle

from qiskit.circuit import QuantumCircuit, QuantumRegister

logger = logging.getLogger(__name__)


def _apply_cu1(circuit, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        circuit.u1(lam / 2, control)
        circuit.cx(control, target)
        circuit.u1(-lam / 2, target)
        circuit.cx(control, target)
        circuit.u1(lam / 2, target)
    else:
        circuit.cu1(lam, control, target)


def _apply_mcu1(circuit, lam, ctls, tgt, global_phase=0):
    """Apply multi-controlled u1 gate from ctls to tgt with angle theta."""

    n = len(ctls)

    from sympy.combinatorics.graycode import GrayCode
    gray_code = list(GrayCode(n).generate_gray())
    last_pattern = None

    lam_angle = lam*(1/(2**(n-1)))
    gp_angle = angle(global_phase)*(1/(2**(n-1)))

    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index('1')

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        # check parity
        if pattern.count('1') % 2 == 0:
            # inverse
            _apply_cu1(circuit, -lam_angle, ctls[lm_pos], tgt)
            if global_phase:
                circuit.u1(-gp_angle, ctls[lm_pos])
        else:
            _apply_cu1(circuit, lam_angle, ctls[lm_pos], tgt)
            if global_phase:
                circuit.u1(gp_angle, ctls[lm_pos])
        last_pattern = pattern


def mcu1(self, lam, control_qubits, target_qubit):
    """
    Apply Multiple-Controlled U1 gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcu1 gate on.
        lam (float): angle lambda
        control_qubits (list(Qubit)): The list of control qubits
        target_qubit (Qubit): The target qubit
    """
    if isinstance(target_qubit, QuantumRegister) and len(target_qubit) == 1:
        target_qubit = target_qubit[0]
    temp = []

    self._check_qargs(control_qubits)
    temp += control_qubits

    self._check_qargs([target_qubit])
    temp.append(target_qubit)

    self._check_dups(temp)
    n_c = len(control_qubits)
    if n_c == 1:  # cu1
        _apply_cu1(self, lam, control_qubits[0], target_qubit)
    else:
        _apply_mcu1(self, lam, control_qubits, target_qubit)


QuantumCircuit.mcu1 = mcu1
