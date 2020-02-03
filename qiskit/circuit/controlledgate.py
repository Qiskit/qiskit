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
                    self.base_gate = base_gate

    def __eq__(self, other):
        if not isinstance(other, ControlledGate):
            return False
        else:
            return (other.num_ctrl_qubits == self.num_ctrl_qubits and
                    self.base_gate == other.base_gate and
                    super().__eq__(other))

    def inverse(self):
        """Invert this gate by calling inverse on the base gate."""
        return self.base_gate.inverse().control(self.num_ctrl_qubits)
