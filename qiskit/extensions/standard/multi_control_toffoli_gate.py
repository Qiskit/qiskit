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
Multiple-Control Toffoli Gate.
"""

import logging
from math import pi, ceil

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit

from qiskit import QiskitError
# pylint: disable=unused-import
from .relative_phase_toffoli import rccx

logger = logging.getLogger(__name__)


def _mct_v_chain(qc, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=False):
    """
    Create new MCT circuit by chaining Toffoli gates into a V shape.

    The dirty_ancilla mode is from https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2

    All intermediate Toffoli gates are implemented up to a relative phase,
    see https://arxiv.org/abs/1508.03273

    An additional saving of 4 CNOTs is achieved
    by using the Toffoli implementation from Section IV.B of https://arxiv.org/abs/1508.03273
    """

    if len(ancillary_qubits) < len(control_qubits) - 2:
        raise QiskitError('Insufficient number of ancillary qubits.')

    if dirty_ancilla:
        anci_idx = len(control_qubits) - 3

        qc.u2(0, pi, target_qubit)
        qc.cx(target_qubit, ancillary_qubits[anci_idx])
        qc.u1(-pi/4, ancillary_qubits[anci_idx])
        qc.cx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx])
        qc.u1(pi/4, ancillary_qubits[anci_idx])
        qc.cx(target_qubit, ancillary_qubits[anci_idx])
        qc.u1(-pi/4, ancillary_qubits[anci_idx])
        qc.cx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx])
        qc.u1(pi/4, ancillary_qubits[anci_idx])

        for idx in reversed(range(2, len(control_qubits) - 1)):
            qc.rccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
            anci_idx -= 1

    anci_idx = 0
    qc.rccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
    for idx in range(2, len(control_qubits) - 1):
        qc.rccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
        anci_idx += 1

    if dirty_ancilla:
        qc.u1(-pi/4, ancillary_qubits[anci_idx])
        qc.cx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx])
        qc.u1(pi/4, ancillary_qubits[anci_idx])
        qc.cx(target_qubit, ancillary_qubits[anci_idx])
        qc.u1(-pi/4, ancillary_qubits[anci_idx])
        qc.cx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx])
        qc.u1(pi/4, ancillary_qubits[anci_idx])
        qc.cx(target_qubit, ancillary_qubits[anci_idx])
        qc.u2(0, pi, target_qubit)
    else:
        qc.ccx(control_qubits[len(control_qubits) - 1], ancillary_qubits[anci_idx], target_qubit)

    for idx in reversed(range(2, len(control_qubits) - 1)):
        qc.rccx(control_qubits[idx], ancillary_qubits[anci_idx - 1], ancillary_qubits[anci_idx])
        anci_idx -= 1
    qc.rccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])

    if dirty_ancilla:
        anci_idx = 0
        for idx in range(2, len(control_qubits) - 1):
            qc.rccx(control_qubits[idx], ancillary_qubits[anci_idx], ancillary_qubits[anci_idx + 1])
            anci_idx += 1


def _cccx(qc, qrs, angle=pi / 4):
    """
    A 3-qubit controlled-NOT.

    Implementation based on Page 17 of Barenco et al.

    Args:
        qc (QuantumCircuit): quantum circuit to apply operation to.
        qrs (list): list of quantum registers. The last qubit is the target,
            the rest are controls
        angle (float) : default pi/4 when x is the NOT gate, set to pi/8 for
            square root of NOT
    """
    assert len(qrs) == 4, "There must be exactly 4 qubits of quantum registers for cccx"

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[0], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[0], qrs[1])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[1], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[1])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[1], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ------------

    qc.cx(qrs[1], qrs[2])

    # controlled-Vdag
    qc.h(qrs[3])
    qc.cu1(angle, qrs[2], qrs[3])
    qc.h(qrs[3])
    # ---------------

    qc.cx(qrs[0], qrs[2])

    # controlled-V
    qc.h(qrs[3])
    qc.cu1(-angle, qrs[2], qrs[3])
    qc.h(qrs[3])


def _ccccx(qc, qrs):
    """
    a 4-qubit controlled-NOT.

    An implementation based on Page 21 (Lemma 7.5) of Barenco et al.

    Args:
        qc (QuantumCircuit): quantum circuit to apply operation to.
        qrs (list): list of quantum registers. The last qubit is the target,
            the rest are controls
    """
    assert len(qrs) == 5, "There must be exactly 5 qubits for ccccx"

    # controlled-V
    qc.h(qrs[4])
    qc.cu1(-pi / 2, qrs[3], qrs[4])
    qc.h(qrs[4])
    # ------------

    _cccx(qc, qrs[:4])

    # controlled-Vdag
    qc.h(qrs[4])
    qc.cu1(pi / 2, qrs[3], qrs[4])
    qc.h(qrs[4])
    # ------------

    _cccx(qc, qrs[:4])
    _cccx(qc, [qrs[0], qrs[1], qrs[2], qrs[4]], angle=pi / 8)


def _multicx(qc, qrs, qancilla=None):
    """
    Construct a circuit for multi-qubit controlled-not

    Args:
        qc (QuantumCircuit): quantum circuit
        qrs (list(QuantumRegister)): list of quantum registers of at least length 1
        qancilla (QuantumRegister): a quantum register. can be None if len(qrs) <= 5
    """
    if not qrs:
        pass
    elif len(qrs) == 1:
        qc.x(qrs[0])
    elif len(qrs) == 2:
        qc.cx(qrs[0], qrs[1])
    elif len(qrs) == 3:
        qc.ccx(qrs[0], qrs[1], qrs[2])
    else:
        _multicx_recursion(qc, qrs, qancilla)


def _multicx_recursion(qc, qrs, qancilla=None):
    if len(qrs) == 4:
        _cccx(qc, qrs)
    elif len(qrs) == 5:
        _ccccx(qc, qrs)
    else:  # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target, and qancilla as working qubit
        assert qancilla is not None, ('There must be an ancilla qubit not '
                                      'necessarily initialized to zero')
        n = len(qrs)
        mid = ceil(n / 2)
        _multicx_recursion(qc, [*qrs[:mid], qancilla], qrs[mid])
        _multicx_recursion(qc, [*qrs[mid:n - 1], qancilla, qrs[n - 1]], qrs[mid - 1])
        _multicx_recursion(qc, [*qrs[:mid], qancilla], qrs[mid])
        _multicx_recursion(qc, [*qrs[mid:n - 1], qancilla, qrs[n - 1]], qrs[mid - 1])


def _multicx_noancilla(qc, qrs):
    """
    Construct a circuit for multi-qubit controlled-not without ancillary qubits

    Args:
        qc (QuantumCircuit): quantum circuit
        qrs (list(QuantumRegister)): list of quantum registers of at least length 1
    """
    if not qrs:
        pass
    elif len(qrs) == 1:
        qc.x(qrs[0])
    elif len(qrs) == 2:
        qc.cx(qrs[0], qrs[1])
    else:
        # qrs[0], qrs[n-2] is the controls, qrs[n-1] is the target
        ctls = qrs[:-1]
        tgt = qrs[-1]
        qc.h(tgt)
        qc.mcu1(pi, ctls, tgt)
        qc.h(tgt)


def mct(self, q_controls, q_target, q_ancilla, mode='basic'):
    """
    Apply Multiple-Control Toffoli operation

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mct gate on.
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancilla (QuantumRegister or list(Qubit)): The list of ancillary qubits
        mode (str): The implementation mode to use

    Raises:
        QiskitError: improper arguments
    """

    if len(q_controls) == 1:  # cx
        self.cx(q_controls[0], q_target)
    elif len(q_controls) == 2:  # ccx
        self.ccx(q_controls[0], q_controls[1], q_target)
    else:
        # check controls
        if isinstance(q_controls, QuantumRegister):
            control_qubits = list(q_controls)
        elif isinstance(q_controls, list):
            control_qubits = q_controls
        else:
            raise QiskitError('MCT needs a list of qubits or a quantum register for controls.')

        # check target
        if isinstance(q_target, Qubit):
            target_qubit = q_target
        else:
            raise QiskitError('MCT needs a single qubit as target.')

        # check ancilla
        if q_ancilla is None:
            ancillary_qubits = []
        elif isinstance(q_ancilla, QuantumRegister):
            ancillary_qubits = list(q_ancilla)
        elif isinstance(q_ancilla, list):
            ancillary_qubits = q_ancilla
        else:
            raise QiskitError('MCT needs None or a list of qubits or a quantum '
                              'register for ancilla.')

        all_qubits = control_qubits + [target_qubit] + ancillary_qubits

        self._check_qargs(all_qubits)
        self._check_dups(all_qubits)

        if mode == 'basic' and q_ancilla:
            _mct_v_chain(self, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=False)
        elif mode == 'basic-dirty-ancilla':
            _mct_v_chain(self, control_qubits, target_qubit, ancillary_qubits, dirty_ancilla=True)
        elif mode == 'advanced' and q_ancilla:
            _multicx(self, [*control_qubits, target_qubit], ancillary_qubits[0]
                     if ancillary_qubits else None)
        elif mode == 'noancilla' or not q_ancilla:
            _multicx_noancilla(self, [*control_qubits, target_qubit])
        else:
            raise QiskitError('Unrecognized mode for building MCT circuit: {}.'.format(mode))


QuantumCircuit.mct = mct
