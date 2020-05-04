# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Controlled unitary gate."""

from typing import Tuple, List, Optional, Union
from qiskit.circuit.exceptions import CircuitError

from .gate import Gate
from .quantumregister import QuantumRegister
from .quantumregister import Qubit
from .classicalregister import Clbit

# pylint: disable=missing-return-doc


class ControlledGate(Gate):
    """Controlled unitary gate."""

    def __init__(self, name: str, num_qubits: int, params: List,
                 label: Optional[str] = None, num_ctrl_qubits: Optional[int] = 1,
                 definition: Optional[List[Tuple[Gate, List[Qubit], List[Clbit]]]] = None,
                 ctrl_state: Optional[Union[int, str]] = None):
        """Create a new ControlledGate. In the new gate the first ``num_ctrl_qubits``
        of the gate are the controls.

        Args:
            name: The name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters for the gate.
            label: An optional label for the gate.
            num_ctrl_qubits: Number of control qubits.
            definition: A list of gate rules for implementing this gate. The
                elements of the list are tuples of (:meth:`~qiskit.circuit.Gate`, [qubit_list],
                [clbit_list]).
            ctrl_state: The control state in decimal or as
                a bitstring (e.g. '111'). If specified as a bitstring the length
                must equal num_ctrl_qubits, MSB on left. If None, use
                2**num_ctrl_qubits-1.

        Raises:
            CircuitError: If ``num_ctrl_qubits`` >= ``num_qubits``.
            CircuitError: ctrl_state < 0 or ctrl_state > 2**num_ctrl_qubits.

        Examples:

        Create a controlled standard gate and apply it to a circuit.

        .. jupyter-execute::

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.circuit.library.standard_gates import HGate

           qr = QuantumRegister(3)
           qc = QuantumCircuit(qr)
           c3h_gate = HGate().control(2)
           qc.append(c3h_gate, qr)
           qc.draw()

        Create a controlled custom gate and apply it to a circuit.

        .. jupyter-execute::

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.circuit.library.standard_gates import HGate

           qc1 = QuantumCircuit(2)
           qc1.x(0)
           qc1.h(1)
           custom = qc1.to_gate().control(2)

           qc2 = QuantumCircuit(4)
           qc2.append(custom, [0, 3, 1, 2])
           qc2.draw()
        """
        super().__init__(name, num_qubits, params, label=label)
        if num_ctrl_qubits < num_qubits:
            self.num_ctrl_qubits = num_ctrl_qubits
        else:
            raise CircuitError('number of control qubits must be less than the number of qubits')
        self.base_gate = None
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
    def definition(self) -> List:
        """Return definition in terms of other basic gates. If the gate has
        open controls, as determined from `self.ctrl_state`, the returned
        definition is conjugated with X without changing the internal
        `_definition`.
        """
        if not self._definition:
            self._define()
        # pylint: disable=cyclic-import
        from qiskit.circuit.library.standard_gates import XGate, CXGate
        bit_ctrl_state = bin(self.ctrl_state)[2:].zfill(self.num_ctrl_qubits)
        # hacky way to get register assuming single register
        if self._definition:
            qreg = self._definition[0][1][0].register
            definition = self._definition
        elif isinstance(self, CXGate):
            qreg = QuantumRegister(self.num_qubits, 'q')
            definition = [(self, [qreg[0], qreg[1]], [])]
        open_rules = []
        for qind, val in enumerate(bit_ctrl_state[::-1]):
            if val == '0':
                open_rules.append([XGate(), [qreg[qind]], []])
        if open_rules:
            return open_rules + definition + open_rules
        else:
            return self._definition

    @definition.setter
    def definition(self, excited_def: List):
        """Set controlled gate definition with closed controls."""
        super(Gate, self.__class__).definition.fset(self, excited_def)

    @property
    def ctrl_state(self) -> int:
        """Return the control state of the gate as a decimal integer."""
        return self._ctrl_state

    @ctrl_state.setter
    def ctrl_state(self, ctrl_state: Union[int, str, None]):
        """Set the control state of this gate.

        Args:
            ctrl_state: The control state of the gate.

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
            raise CircuitError('invalid control state specification: {}'.format(
                repr(ctrl_state)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, ControlledGate) and
                self.num_ctrl_qubits == other.num_ctrl_qubits and
                self.ctrl_state == other.ctrl_state and
                self.base_gate == other.base_gate and
                self.num_qubits == other.num_qubits and
                self.num_clbits == other.num_clbits and
                self.definition == other.definition)

    def inverse(self) -> 'ControlledGate':
        """Invert this gate by calling inverse on the base gate."""
        return self.base_gate.inverse().control(self.num_ctrl_qubits)
