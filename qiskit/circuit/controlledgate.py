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

from qiskit.exceptions import QiskitError
from .instruction import Instruction
import qiskit.circuit.gate as gate


class ControlledGate(gate.Gate):
    """Controlled unitary gate."""

    def __init__(self,  name, num_qubits, params, label=None, num_ctrl_qubits=1,
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
            QiskitError: num_ctrl_qubits < num_qubits
        """
        super().__init__(name, num_qubits, params, label=label)
        if num_ctrl_qubits < num_qubits:
            self.num_ctrl_qubits = num_ctrl_qubits
        else:
            raise QiskitError('number of control qubits must be less than the number of qubits')
        if definition:
            self.definition = definition


    def __eq__(self, other):
        if not isinstance(other, ControlledGate):
            return False
        else:
            return (other.num_ctrl_qubits == self.num_ctrl_qubits and
                    super().__eq__(other))
