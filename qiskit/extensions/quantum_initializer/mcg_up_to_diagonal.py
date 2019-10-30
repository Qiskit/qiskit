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

# pylint: disable=unused-variable
# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Multi controlled single-qubit unitary up to diagonal.
"""

# ToDo: This code should be merged wth the implementation of MCGs
# ToDo: (introducing a decomposition mode "up_to_diagonal").

import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info.operators.predicates import is_isometry
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer.ucg import UCG

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class MCGupDiag(Gate):
    """
    Decomposes a multi-controlled gate u up to a diagonal d acting on the control and target qubit
    (but not on the  ancilla qubits), i.e., it implements a circuit corresponding to a unitary u'
    such that u=d.u'.
    """

    def __init__(self, gate, num_controls, num_ancillas_zero, num_ancillas_dirty):
        """Initialize a multi controlled gate.

            Args:
                gate (ndarray): 2*2 unitary (given as a (complex) ndarray)
                num_controls (int): number of control qubits
                num_ancillas_zero (int): number of ancilla qubits that start in the state zero
                num_ancillas_dirty (int): number of anxilla qubits that are allowed to start in an
                    arbitrary state
            Raises:
                QiskitError: if the input format is wrong; if the array gate is not unitary
        """

        self.num_controls = num_controls
        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        # Check if the gate has the right dimension
        if not gate.shape == (2, 2):
            raise QiskitError("The dimension of the controlled gate is not equal to (2,2).")
        # Check if the single-qubit gate is unitary
        if not is_isometry(gate, _EPS):
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
    def _get_diagonal(self):
        # Important: for a control list q_controls = [q[0],...,q_[k-1]] the diagonal gate is
        # provided in the computational basis of the qubits q[k-1],...,q[0],q_target, decreasingly
        # ordered with respect to the significance of the qubit in the computational basis
        _, diag = self._dec_mcg_up_diag()
        return diag

    def _dec_mcg_up_diag(self):
        """
        Call to create a circuit with gates that implement the MCG up to a diagonal gate.
        Remark: The qubits the gate acts on are ordered in the following way:
            q=[q_target,q_controls,q_ancilla_zero,q_ancilla_dirty]
        """
        diag = np.ones(2 ** (self.num_controls + 1)).tolist()
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        (q_target, q_controls, q_ancillas_zero, q_ancillas_dirty) = self._define_qubit_role(q)
        # ToDo: Keep this threshold updated such that the lowest gate count is achieved:
        # ToDo: we implement the MCG with a UCG up to diagonal if the number of controls is
        # ToDo: smaller than the threshold.
        threshold = float("inf")
        if self.num_controls < threshold:
            # Implement the MCG as a UCG (up to diagonal)
            gate_list = [np.eye(2, 2) for i in range(2 ** self.num_controls)]
            gate_list[-1] = self.params[0]
            ucg = UCG(gate_list, up_to_diagonal=True)
            circuit.append(ucg, [q_target] + q_controls)
            diag = ucg._get_diagonal()
            # else:
            # ToDo: Use the best decomposition for MCGs up to diagonal gates here
            # ToDo: (with all available ancillas)
        return circuit, diag

    def _define_qubit_role(self, q):
        # Define the role of the qubits
        q_target = q[0]
        q_controls = q[1:self.num_controls + 1]
        q_ancillas_zero = q[self.num_controls + 1:self.num_controls + 1 + self.num_ancillas_zero]
        q_ancillas_dirty = q[self.num_controls + 1 + self.num_ancillas_zero:]
        return q_target, q_controls, q_ancillas_zero, q_ancillas_dirty
