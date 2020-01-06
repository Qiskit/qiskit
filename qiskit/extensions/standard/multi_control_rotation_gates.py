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
Multiple-Controlled U3 gate. Not using ancillary qubits.
"""

import logging
from math import pi
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit import QiskitError

logger = logging.getLogger(__name__)


def _apply_cu3(circuit, theta, phi, lam, control, target, use_basis_gates=True):
    if use_basis_gates:
        circuit.u1((lam + phi) / 2, control)
        circuit.u1((lam - phi) / 2, target)
        circuit.cx(control, target)
        circuit.u3(-theta / 2, 0, -(phi + lam) / 2, target)
        circuit.cx(control, target)
        circuit.u3(theta / 2, phi, 0, target)
    else:
        circuit.cu3(theta, phi, lam, control, target)


def _apply_mcu3_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u3 gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""

    n = len(ctls)

    from sympy.combinatorics.graycode import GrayCode
    gray_code = list(GrayCode(n).generate_gray())
    last_pattern = None

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
        # check parity and undo rotation
        if pattern.count('1') % 2 == 0:
            # inverse CU3: u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
            _apply_cu3(circuit, -theta, -lam, -phi, ctls[lm_pos], tgt,
                       use_basis_gates=use_basis_gates)
        else:
            _apply_cu3(circuit, theta, phi, lam, ctls[lm_pos], tgt,
                       use_basis_gates=use_basis_gates)
        last_pattern = pattern


def mcrx(self, theta, q_controls, q_target, use_basis_gates=False):
    """
    Apply Multiple-Controlled X rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrx gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use u1, u2, u3, cx, id

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError(
            'The mcrx gate needs a list of qubits or a quantum register for controls.')

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError('The mcrx gate needs a single qubit as target.')

    all_qubits = control_qubits + [target_qubit]

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu3
        _apply_cu3(self, theta, -pi/2, pi/2, control_qubits[0],
                   target_qubit, use_basis_gates=use_basis_gates)
    else:
        theta_step = theta * (1 / (2 ** (n_c - 1)))
        _apply_mcu3_graycode(self, theta_step, -pi/2, pi/2, control_qubits,
                             target_qubit, use_basis_gates=use_basis_gates)


def mcry(self, theta, q_controls, q_target, q_ancillae, mode='basic',
         use_basis_gates=False):
    """
    Apply Multiple-Controlled Y rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcry gate on.
        theta (float): angle theta
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancillae (QuantumRegister or tuple(QuantumRegister, int)): The list of ancillary qubits.
        mode (string): The implementation mode to use
        use_basis_gates (bool): use u1, u2, u3, cx, id

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError('The mcry gate needs a list of qubits or a quantum '
                          'register for controls.')

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError('The mcry gate needs a single qubit as target.')

    # check ancilla
    if q_ancillae is None:
        ancillary_qubits = []
    elif isinstance(q_ancillae, QuantumRegister):
        ancillary_qubits = list(q_ancillae)
    elif isinstance(q_ancillae, list):
        ancillary_qubits = q_ancillae
    else:
        raise QiskitError('The mcry gate needs None or a list of qubits or a '
                          'quantum register for ancilla.')

    all_qubits = control_qubits + [target_qubit] + ancillary_qubits

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    if mode == 'basic':
        self.u3(theta / 2, 0, 0, q_target)
        self.mct(q_controls, q_target, q_ancillae)
        self.u3(-theta / 2, 0, 0, q_target)
        self.mct(q_controls, q_target, q_ancillae)
    elif mode == 'noancilla':
        n_c = len(control_qubits)
        if n_c == 1:  # cu3
            _apply_cu3(self, theta, 0, 0, control_qubits[0],
                       target_qubit, use_basis_gates=use_basis_gates)
        else:
            theta_step = theta * (1 / (2 ** (n_c - 1)))
            _apply_mcu3_graycode(self, theta_step, 0, 0, control_qubits,
                                 target_qubit, use_basis_gates=use_basis_gates)
    else:
        raise QiskitError('Unrecognized mode for building MCRY circuit: {}.'.format(mode))


def mcrz(self, lam, q_controls, q_target, use_basis_gates=False):
    """
    Apply Multiple-Controlled Z rotation gate

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcrz gate on.
        lam (float): angle lambda
        q_controls (list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        use_basis_gates (bool): use u1, u2, u3, cx, id

    Raises:
        QiskitError: parameter errors
    """

    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise QiskitError(
            'The mcrz gate needs a list of qubits or a quantum register for controls.')

    # check target
    if isinstance(q_target, Qubit):
        target_qubit = q_target
    else:
        raise QiskitError('The mcrz gate needs a single qubit as target.')

    all_qubits = control_qubits + [target_qubit]

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    n_c = len(control_qubits)
    if n_c == 1:  # cu3
        _apply_cu3(self, 0, 0, lam, control_qubits[0],
                   target_qubit, use_basis_gates=use_basis_gates)
    else:
        lam_step = lam * (1 / (2 ** (n_c - 1)))
        _apply_mcu3_graycode(self, 0, 0, lam_step, control_qubits,
                             target_qubit, use_basis_gates=use_basis_gates)


QuantumCircuit.mcrx = mcrx
QuantumCircuit.mcry = mcry
QuantumCircuit.mcrz = mcrz
