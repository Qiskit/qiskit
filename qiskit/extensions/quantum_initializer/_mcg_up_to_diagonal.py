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

from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer import UCG

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class MCGupDiag(Gate):
    """
    Decomposes a multi-controlled gate u up to a diagonal d acting on the control and target qubit (but not on the
    ancilla qubits), i.e., it implements a circuit corresponding to a unitary u' such that u=d.u'.
    
    Input:
    gate =  a single-qubit unitary U  a given as a 2*2 numpy array.

    num_ancillas_zero = number of additional ancillas that start in the state ket(0) (the n-m ancillas required
                        for providing the ouput of the isometry are not accounted for here).

    num_ancillas_dirty = number of additional ancillas that start in an arbitrary state
    """

    """
    Remark: The qubits the gate acts on should be provided in the following order: 
    q=[q_target,q_controls,q_ancilla_zero,q_ancilla_dirty], where

    q_controls =   list of k control qubits.
                    [The qubits are ordered with increasing significance (this will
                    determine the basis in which the diagonal gate is stored in)]

    q_target =     target qubit, where we act on with the single-qubit gates.
    """

    def __init__(self, gate, num_controls, num_ancillas_zero, num_ancillas_dirty):
        self.num_controls = num_controls
        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        # Check if the gate has the right dimension
        if not gate.shape == (2, 2):
            raise QiskitError("The dimension of the controlled gate is not equal to (2,2).")

        """Check if the input has the correct form"""
        # Check if the single-qubit gate is unitary
        if not _is_isometry(gate, _EPS):
            raise QiskitError("The controlled gate is not unitary.")

        # Create new gate.
        num_qubits = 1 + num_controls + num_ancillas_zero + num_ancillas_dirty
        super().__init__("MCGupDiag", num_qubits, [gate])

    def _define(self):
        mcg_up_diag_circuit, _ = self._dec_mcg_up_diag()
        gate = mcg_up_diag_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        mcg_up_diag_circuit = QuantumCircuit(q)
        mcg_up_diag_circuit.append(gate, q[:])
        self.definition = mcg_up_diag_circuit.data

    # Returns the diagonal up to which the gate is implemented.
    def get_diagonal(self):
        # Important: for a control list q_controls = [q[0],...,q_[k-1]] the diagonal gate is provided in the
        # computational basis of the qubits q[k-1],...,q[0],q_target, decreasingly ordered with respect to the
        # significance of the qubit in the computational basis
        _, diag = self._dec_mcg_up_diag()
        return diag

    def _dec_mcg_up_diag(self):
        """
        Call to create a circuit with gates that implement the MCG up to a diagonal gate.
        """
        diag = np.ones(2**(self.num_controls+1)).tolist()
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        (q_target, q_controls, q_ancillas_zero, q_ancillas_dirty) = self._define_qubit_role(q)
        # ToDo: Keep this threshold updated such that the lowest gate count is achieved:  we implement
        # ToDo: the MCG with a UCG up to diagonal if the number of controls is smaller than the threshold.
        threshold = float("inf")
        if self.num_controls < threshold:
            # Implement the MCG as a UCG (up to diagonal)
            gate_list = [np.eye(2, 2) for i in range(2 ** self.num_controls)]
            gate_list[-1] = self.params[0]
            ucg = UCG(gate_list, up_to_diagonal=True)
            circuit.append(ucg, [q_target] + q_controls)
            diag = ucg.get_diagonal()
        else:
            # ToDo: Use the best decomposition for MCGs up to diagonal gates here (with all available ancillas)
            None
        return circuit, diag

    def _define_qubit_role(self, q):
        # Define the role of the qubits
        q_target = q[0]
        q_controls = q[1:self.num_controls+1]
        q_ancillas_zero = q[self.num_controls+1:self.num_controls+1+self.num_ancillas_zero]
        q_ancillas_dirty = q[self.num_controls+1+self.num_ancillas_zero:]
        return q_target, q_controls, q_ancillas_zero, q_ancillas_dirty


def _is_isometry(m, eps):
    err = np.linalg.norm(np.dot(np.transpose(np.conj(m)), m) - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=eps)



