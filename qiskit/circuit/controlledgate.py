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
Controlled unitary gate.
"""

import numpy as np
from qiskit.circuit.exceptions import CircuitError
from .gate import Gate


class ControlledGate(Gate):
    """Controlled unitary gate."""

    def __init__(self, name, num_qubits, params, label=None, num_ctrl_qubits=1,
                 definition=None):
        """Create a new gate.

        Args:
            name (str): The Qobj name of the gate.
            num_qubits (int): The number of qubits the gate acts on.
            params (list): A list of parameters.
            label (str or None): An optional label for the gate [Default: None]
            num_ctrl_qubits (int): Number of control qubits.
            definition (list): list of gate rules for implementing this gate.
        Raises:
            CircuitError: num_ctrl_qubits >= num_qubits
        """
        super().__init__(name, num_qubits, params, label=label)
        if num_ctrl_qubits < num_qubits:
            self.num_ctrl_qubits = num_ctrl_qubits
        else:
            raise CircuitError('number of control qubits must be less than the number of qubits')
        if definition:
            self.definition = definition
            if len(definition) == 1:
                base_gate = definition[0][0]
                if isinstance(base_gate, ControlledGate):
                    self.base_gate = base_gate.base_gate
                else:
                    self.base_gate = base_gate.__class__

                self.base_gate_name = base_gate.name

    def to_matrix(self, phase=0):
        """
        Return matrix form of controlled gate.

        Args:
            base_mat (ndarray): unitary to be controlled
            num_ctrl_qubits (int): number of controls for new unitary
            phase (float): The global phase of base_mat which is promoted to the
                global phase of the controlled matrix

        Returns:
            ndarray: controlled version of base matrix.
        """
        try:
            # for standard extension gates
            base_mat = self.base_gate(*self.params).to_matrix()
        except:
            if isinstance(self.base_gate, ControlledGate):
                base_mat = self.base_gate(self.name, self.num_qubits,
                                          self.params, label=self.label,
                                          num_ctrl_qubits=self.num_ctrl_qubits)
            else:
                base_mat = self.base_gate(self.name, self.num_qubits,
                                          self.params, label=self.label)
        #import ipdb;ipdb.set_trace()


        num_ctrl_qubits = self.num_ctrl_qubits
        num_target = int(np.log2(base_mat.shape[0]))
        ctrl_dim = 2**num_ctrl_qubits
        ctrl_grnd = np.repeat([[1], [0]], [1, ctrl_dim-1])
        full_mat_dim = ctrl_dim * base_mat.shape[0]
        full_mat = np.zeros((full_mat_dim, full_mat_dim), dtype=base_mat.dtype)
        for i in range(ctrl_dim-1):
            full_mat += np.kron(np.eye(2**num_target),
                                np.diag(np.roll(ctrl_grnd, i)))
        if phase != 0:
            full_mat = np.exp(1j * phase) * full_mat
        full_mat += np.kron(base_mat, np.diag(np.roll(ctrl_grnd, ctrl_dim-1)))
        return full_mat
        
    def __eq__(self, other):
        if not isinstance(other, ControlledGate):
            return False
        else:
            return (other.num_ctrl_qubits == self.num_ctrl_qubits and
                    self.base_gate == other.base_gate and
                    super().__eq__(other))
