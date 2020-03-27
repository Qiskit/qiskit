# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multiple-Control, Multiple-Target Gate.
"""

import logging

from qiskit.circuit import Gate  # pylint: disable=unused-import
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit  # pylint: disable=unused-import

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


def _ccx_v_chain_compute(qc, control_qubits, ancillary_qubits):
    """
    First half (compute) of the multi-control basic mode. It progressively compute the
    ccx of the control qubits and put the final result in the last ancillary
    qubit
    Args:
        qc (QuantumCircuit): the Quantum Circuit
        control_qubits (list): the list of control qubits
        ancillary_qubits (list): the list of ancillary qubits

    """
    anci_idx = 0
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])
    for idx in range(2, len(control_qubits)):
        assert anci_idx + 1 < len(
            ancillary_qubits
        ), "Insufficient number of ancillary qubits {0}.".format(
            len(ancillary_qubits))
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx],
               ancillary_qubits[anci_idx + 1])
        anci_idx += 1


def _ccx_v_chain_uncompute(qc, control_qubits, ancillary_qubits):
    """
    Second half (uncompute) of the multi-control basic mode. It progressively compute the
    ccx of the control qubits and put the final result in the last ancillary
    qubit
    Args:
        qc (QuantumCircuit): the Quantum Circuit
        control_qubits (list): the list of control qubits
        ancillary_qubits (list): the list of ancillary qubits
    """
    anci_idx = len(ancillary_qubits) - 1
    for idx in (range(2, len(control_qubits)))[::-1]:
        qc.ccx(control_qubits[idx], ancillary_qubits[anci_idx - 1],
               ancillary_qubits[anci_idx])
        anci_idx -= 1
    qc.ccx(control_qubits[0], control_qubits[1], ancillary_qubits[anci_idx])


def mcmt(self,
         q_controls,
         q_ancillae,
         single_control_gate_fun,
         q_targets,
         mode="basic"):
    """
    Apply a Multi-Control, Multi-Target using a generic gate.
    It can also be used to implement a generic Multi-Control gate,
    as the target could also be of length 1.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to apply the mcmt gate on.
        q_controls (Union(QuantumRegister, list[Qubit])): The list of control qubits
        q_ancillae (Union(QuantumRegister, list[Qubit])): The list of ancillary qubits
        single_control_gate_fun (Gate): The single control gate function (e.g QuantumCircuit.cz
                                        or QuantumCircuit.ch)
        q_targets (Union(QuantumRegister, list[Qubit])): A list of qubits or a QuantumRegister
            to which the gate function should be applied.
        mode (str): The implementation mode to use (at the moment, only the basic mode is supported)
    Raises:
        AquaError: invalid input

    """
    # check controls
    if isinstance(q_controls, QuantumRegister):
        control_qubits = list(q_controls)
    elif isinstance(q_controls, list):
        control_qubits = q_controls
    else:
        raise AquaError('MCT needs a list of qubits or a quantum register for controls.')

    # check target
    if isinstance(q_targets, QuantumRegister):
        target_qubits = list(q_targets)
    elif isinstance(q_targets, list):
        target_qubits = q_targets
    else:
        raise AquaError('MCT needs a list of qubits or a quantum register for targets.')

    # check ancilla
    if q_ancillae is None:
        ancillary_qubits = []
    elif isinstance(q_ancillae, QuantumRegister):
        ancillary_qubits = list(q_ancillae)
    elif isinstance(q_ancillae, list):
        ancillary_qubits = q_ancillae
    else:
        raise AquaError('MCT needs None or a list of qubits or a quantum register for ancilla.')

    all_qubits = control_qubits + target_qubits + ancillary_qubits

    self._check_qargs(all_qubits)
    self._check_dups(all_qubits)

    if len(q_controls) == 1:
        for qubit in target_qubits:
            single_control_gate_fun(self, q_controls[0], qubit)
        return

    if mode == 'basic':
        # last ancillary qubit is the control of the gate
        ancn = len(ancillary_qubits)
        _ccx_v_chain_compute(self, control_qubits, ancillary_qubits)
        for qubit in target_qubits:
            single_control_gate_fun(self, ancillary_qubits[ancn - 1], qubit)
        _ccx_v_chain_uncompute(self, control_qubits, ancillary_qubits)
    else:
        raise AquaError(
            'Unrecognized mode "{0}" for building mcmt circuit, '
            'at the moment only "basic" mode is supported.'
            .format(mode))


QuantumCircuit.mcmt = mcmt
