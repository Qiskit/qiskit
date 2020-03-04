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
from . import QuantumRegister


class ControlledGate(Gate):
    """Controlled unitary gate."""

    def __init__(self, name, num_qubits, params, num_ctrl_qubits=1, phase=0,
                 label=None, definition=None, ctrl_state=None):
        """Create a controlled gate.

        Attributes:
            num_ctrl_qubits (int): The number of control qubits.
            ctrl_state (int): The control state in decimal notation.

        Args:
            name (str): The Qobj name of the gate.
            num_qubits (int): The number of qubits the gate acts on.
            params (list): A list of parameters.
            phase (float): set the gate phase (Default: 0).
            label (str or None): An optional label for the gate (Default: None).
            num_ctrl_qubits (int): Number of control qubits.
            definition (list): list of gate rules for implementing this gate.
            ctrl_state (int or str or None): The control state in decimal or as
                a bitstring (e.g. '111'). If specified as a bitstring the length
                must equal num_ctrl_qubits, MSB on left. If None, use
                2**num_ctrl_qubits-1.
        Raises:
            CircuitError: num_ctrl_qubits >= num_qubits
            CircuitError: ctrl_state < 0 or ctrl_state > 2**num_ctrl_qubits.
        """
        super().__init__(name, num_qubits, params, phase=phase, label=label)
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
        self._ctrl_state = None
        self.ctrl_state = ctrl_state

    @property
    def definition(self):
        """Return definition in terms of other basic gates. If the gate has
        open controls, as determined from `self.ctrl_state`, the returned
        definition is conjugated with X."""
        if not self._definition:
            self._define()
        # pylint: disable=cyclic-import
        from qiskit.extensions.standard import XGate, CnotGate
        bit_ctrl_state = bin(self.ctrl_state)[2:].zfill(self.num_ctrl_qubits)
        # hacky way to get register assuming single register
        if self._definition:
            qreg = self._definition[0][1][0].register
        elif isinstance(self, CnotGate):
            qreg = QuantumRegister(self.num_qubits, 'q')
            self._definition = [(self, [qreg[0], qreg[1]], [])]
        open_rules = []
        for qind, val in enumerate(bit_ctrl_state[::-1]):
            if val == '0':
                open_rules.append([XGate(), [qreg[qind]], []])
        return open_rules + self._definition + open_rules

    @definition.setter
    def definition(self, excited_def):
        """Set controlled gate definition with closed controls."""
        super(Gate, self.__class__).definition.fset(self, excited_def)

    @property
    def ctrl_state(self):
        """Return the control state of the gate as a decimal integer."""
        return self._ctrl_state

    @ctrl_state.setter
    def ctrl_state(self, ctrl_state):
        """Set the control state of this gate.

        Args:
            ctrl_state (int or str or None): The control state of the gate.

        Raises:
            CircuitError: ctrl_state is invalid.
        """
        if isinstance(ctrl_state, str):
            try:
                assert len(ctrl_state) == self.num_ctrl_qubits
                ctrl_state = int(ctrl_state, 2)
            except ValueError:
                raise CircuitError('invalid control bit string: ' + ctrl_state)
            except AssertionError:
                raise CircuitError('invalid control bit string: length != '
                                   'num_ctrl_qubits')
        if isinstance(ctrl_state, int):
            if 0 <= ctrl_state < 2**self.num_ctrl_qubits:
                self._ctrl_state = ctrl_state
            else:
                raise CircuitError('invalid control state specification')
        elif ctrl_state is None:
            self._ctrl_state = 2**self.num_ctrl_qubits - 1
        else:
            raise CircuitError('invalid control state specification')

    def __eq__(self, other):
        if not isinstance(other, ControlledGate):
            return False
        else:
            return (other.num_ctrl_qubits == self.num_ctrl_qubits and
                    self.base_gate == other.base_gate)

    def inverse(self):
        """Invert this gate by calling inverse on the base gate."""
        return self.base_gate.inverse().control(self.num_ctrl_qubits)
