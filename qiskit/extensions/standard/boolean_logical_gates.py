# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
The Boolean Logical AND and OR Gates.
"""

import logging
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.exceptions import CircuitError
from qiskit.qasm import pi


logger = logging.getLogger(__name__)

# pylint: disable=expression-not-assigned


def _logical_and(circuit, variable_register, flags, target_qubit, ancillary_register, mct_mode):
    if flags is not None:
        zvf = list(zip(variable_register, flags))
        ctl_bits = [v for v, f in zvf if f]
        anc_bits = None
        if ancillary_register:
            anc_bits = [ancillary_register[idx] for idx in range(np.count_nonzero(flags) - 2)]

        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f < 0]
        circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f < 0]


def _logical_or(circuit, qr_variables, flags, qb_target, qr_ancillae, mct_mode):
    circuit.u3(pi, 0, pi, qb_target)
    if flags is not None:
        zvf = list(zip(qr_variables, flags))
        ctl_bits = [v for v, f in zvf if f]
        anc_bits = \
            [qr_ancillae[idx] for idx in range(np.count_nonzero(flags) - 2)] \
            if qr_ancillae else None

        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f > 0]
        circuit.mct(ctl_bits, qb_target, anc_bits, mode=mct_mode)
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f > 0]


def _do_checks(flags, qr_variables, qb_target, qr_ancillae, circuit):
    # check flags
    if flags is None:
        flags = [1 for i in range(len(qr_variables))]
    else:
        if len(flags) > len(qr_variables):
            raise CircuitError('`flags` cannot be longer than `qr_variables`.')

    # check variables
    # TODO: improve the check
    if isinstance(qr_variables, (QuantumRegister, list)):
        variable_qubits = [qb for qb, i in zip(qr_variables, flags) if not i == 0]
    else:
        raise ValueError('A QuantumRegister or list of qubits is expected for variables.')

    # check target
    if isinstance(qb_target, Qubit):
        target_qubit = qb_target
    else:
        raise ValueError('A single qubit is expected for the target.')

    # check ancilla
    if qr_ancillae is None:
        ancillary_qubits = []
    elif isinstance(qr_ancillae, QuantumRegister):
        ancillary_qubits = list(qr_ancillae)
    elif isinstance(qr_ancillae, list):
        ancillary_qubits = qr_ancillae
    else:
        raise ValueError('An optional list of qubits or a '
                         'QuantumRegister is expected for ancillae.')

    all_qubits = variable_qubits + [target_qubit] + ancillary_qubits

    circuit._check_qargs(all_qubits)
    circuit._check_dups(all_qubits)

    return flags


def logical_and(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='no-ancilla'):
    """
    Build a collective conjunction (AND) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the conjunction on.
        qr_variables (QuantumRegister): The QuantumRegister holding the variable qubits.
        qb_target (Qubit): The target qubit to hold the conjunction result.
        qr_ancillae (QuantumRegister): The ancillary QuantumRegister for building the mct.
        flags (list[int]): A list of +1/-1/0 to mark negations or omissions of qubits.
        mct_mode (str): The mct building mode.
    """
    flags = _do_checks(flags, qr_variables, qb_target, qr_ancillae, self)
    _logical_and(self, qr_variables, flags, qb_target, qr_ancillae, mct_mode)


def logical_or(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='basic'):
    """
    Build a collective disjunction (OR) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the disjunction on.
        qr_variables (QuantumRegister): The QuantumRegister holding the variable qubits.
        flags (list[int]): A list of +1/-1/0 to mark negations or omissions of qubits.
        qb_target (Qubit): The target qubit to hold the disjunction result.
        qr_ancillae (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """
    flags = _do_checks(flags, qr_variables, qb_target, qr_ancillae, self)
    _logical_or(self, qr_variables, flags, qb_target, qr_ancillae, mct_mode)


QuantumCircuit.AND = logical_and
QuantumCircuit.OR = logical_or
