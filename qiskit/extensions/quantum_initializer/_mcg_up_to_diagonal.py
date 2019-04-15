# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Multi controlled single-qubit unitary up to diagonal.
"""

# ToDo: This code should be merged wth the implementation of MCGs (introducing a decomposition mode "up_to_diagonal").

import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer import UCG
from qiskit.extensions.quantum_initializer.diag import DiagGate
from qiskit.extensions.quantum_initializer.squ import SingleQubitUnitary
from qiskit.extensions.standard.cx import CnotGate
import cmath

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class MCGupDiag(CompositeGate):
    """Multi-controlled gates.
    
    Input:
    gate_list =     a single-qubit unitary U  a given as a 2*2 numpy array.
                    
    q_controls =    list of k control qubits.
                    [The qubits are ordered with increasing significance (this will
                    determine the basis in which the diagonal gate is stored in)]

    q_target =      target qubit, where we act on with the single-qubit gates.
    
    circ =          QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, gate, q_controls, q_target, q_ancillas_zero=[], q_ancillas_dirty=[], circ=None):
        self.q_controls = q_controls
        self.q_target = q_target
        """Check types"""
        if not type(q_controls) == list:
            raise QiskitError(
                "The control qubits must be provided as a list (also if there is only one control qubit).")
        if not type(q_ancillas_zero) == list:
            raise QiskitError("The ancilla qubits starting in the zero state must be provided as a list.")
        if not type(q_ancillas_dirty) == list:
            raise QiskitError("The dirty ancilla qubits must be provided as a list.")
        # Check if the gate has the right dimension
        if not gate.shape == (2, 2):
            raise QiskitError("The dimension of the controlled gate is not equal to (2,2).")

        """Check if the input has the correct form"""
        # Check if the single-qubit gate is unitary
        if not _is_isometry(gate, _EPS):
            raise QiskitError("The controlled gate is not unitary.")

        # Create new composite gate.
        num_qubits = len(q_controls) + 1 + len(q_ancillas_zero) + len(q_ancillas_dirty)
        self.num_qubits = int(num_qubits)
        qubits = [q_target] + q_controls + q_ancillas_zero + q_ancillas_dirty
        # Important: for a control list q_controls = [q[0],...,q_[k-1]] the diagonal gate is provided in the
        # computational basis of the qubits q[k-1],...,q[0],q_target, decreasingly ordered with respect to the
        # significance of the qubit in the computational basis
        self.diag = np.ones(2 ** (len(q_controls)+1))
        super().__init__("init", [gate], qubits, circ)
        # Check if a qubit is listed twice
        self._check_dups(qubits, message="There is a qubit that is listed in two inputs (which is not allowed).")
        # call to generate the circuit that implements the MCG
        self.dec_mcg_up_diag()

    def dec_mcg_up_diag(self):
        """
        Call to populate the self.data list with gates that implement the MCG up to a diagonal gate, which is stored
        in self.diag.
        """
        # ToDo: Keep this threshold updated such that the lowest gate count is achieved:  we implement
        # ToDo: the MCG with a UCG up to diagonal if the number of controls is smaller than the threshold.
        threshold = float("inf")
        if len(self.q_controls) < threshold:
            # Implement the MCG as a UCG (up to diagonal)
            gate_list = [np.eye(2, 2) for i in range(2 ** len(self.q_controls))]
            gate_list[-1] = self.params[0]
            ucg = UCG(gate_list, self.q_controls, self.q_target, up_to_diagonal=True)
            self._attach(ucg)
            self.diag = ucg.diag
        else:
            # ToDo: Use the best decomposition for MCGs up to diagonal gates here (with all available ancillas)
            None


def _is_isometry(m, eps):
    err = np.linalg.norm(np.dot(np.transpose(np.conj(m)), m) - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=eps)
