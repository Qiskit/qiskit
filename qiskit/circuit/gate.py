# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary gate."""

from typing import List, Optional, Union, Tuple
import numpy as np
from scipy.linalg import schur

from qiskit.circuit.exceptions import CircuitError
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name: str, num_qubits: int, params: List,
                 label: Optional[str] = None) -> None:
        """Create a new gate.

        Args:
            name: The Qobj name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
        """
        self._label = label
        self.definition = None
        super().__init__(name, num_qubits, 0, params)

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        raise CircuitError("to_matrix not defined for this {}".format(type(self)))

    def power(self, exponent: float):
        """Creates a unitary gate as `gate^exponent`.

        Args:
            exponent (float): Gate^exponent

        Returns:
            qiskit.extensions.UnitaryGate: To which `to_matrix` is self.to_matrix^exponent.

        Raises:
            CircuitError: If Gate is not unitary
        """
        from qiskit.quantum_info.operators import Operator  # pylint: disable=cyclic-import
        from qiskit.extensions.unitary import UnitaryGate  # pylint: disable=cyclic-import
        # Should be diagonalized because it's a unitary.
        decomposition, unitary = schur(Operator(self).data, output='complex')
        # Raise the diagonal entries to the specified power
        decomposition_power = list()

        decomposition_diagonal = decomposition.diagonal()
        # assert off-diagonal are 0
        if not np.allclose(np.diag(decomposition_diagonal), decomposition):
            raise CircuitError('The matrix is not diagonal')

        for element in decomposition_diagonal:
            decomposition_power.append(pow(element, exponent))
        # Then reconstruct the resulting gate.
        unitary_power = unitary @ np.diag(decomposition_power) @ unitary.conj().T
        return UnitaryGate(unitary_power, label='%s^%s' % (self.name, exponent))

    def _return_repeat(self, exponent: float) -> 'Gate':
        return Gate(name="%s*%s" % (self.name, exponent), num_qubits=self.num_qubits,
                    params=self.params)

    def assemble(self) -> 'Instruction':
        """Assemble a QasmQobjInstruction"""
        instruction = super().assemble()
        if self.label:
            instruction.label = self.label
        return instruction

    @property
    def label(self) -> str:
        """Return gate label"""
        return self._label

    @label.setter
    def label(self, name: str):
        """Set gate label to name

        Args:
            name (str or None): label to assign unitary

        Raises:
            TypeError: name is not string or None.
        """
        if isinstance(name, (str, type(None))):
            self._label = name
        else:
            raise TypeError('label expects a string or None')

    def control(self, num_ctrl_qubits: Optional[int] = 1, label: Optional[str] = None,
                ctrl_state: Optional[Union[int, str]] = None):
        """Return controlled version of gate. See :class:`.ControlledGate` for usage.

        Args:
            num_ctrl_qubits: number of controls to add to gate (default=1)
            label: optional gate label
            ctrl_state: The control state in decimal or as a bitstring
                (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            qiskit.circuit.ControlledGate: Controlled version of gate. This default algorithm
            uses num_ctrl_qubits-1 ancillae qubits so returns a gate of size
            num_qubits + 2*num_ctrl_qubits - 1.

        Raises:
            QiskitError: unrecognized mode or invalid ctrl_state
        """
        # pylint: disable=cyclic-import
        from .add_control import add_control
        return add_control(self, num_ctrl_qubits, label, ctrl_state)

    @staticmethod
    def _broadcast_single_argument(qarg: List) -> List:
        """Expands a single argument.

        For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
        # [q[0], q[1]] -> [q[0]]
        #              -> [q[1]]
        for arg0 in qarg:
            yield [arg0], []

    @staticmethod
    def _broadcast_2_arguments(qarg0: List, qarg1: List) -> List:
        if len(qarg0) == len(qarg1):
            # [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]]
            #                              -> [q[1], r[1]]
            for arg0, arg1 in zip(qarg0, qarg1):
                yield [arg0, arg1], []
        elif len(qarg0) == 1:
            # [[q[0]], [r[0], r[1]]] -> [q[0], r[0]]
            #                        -> [q[0], r[1]]
            for arg1 in qarg1:
                yield [qarg0[0], arg1], []
        elif len(qarg1) == 1:
            # [[q[0], q[1]], [r[0]]] -> [q[0], r[0]]
            #                        -> [q[1], r[0]]
            for arg0 in qarg0:
                yield [arg0, qarg1[0]], []
        else:
            raise CircuitError('Not sure how to combine these two-qubit arguments:\n %s\n %s' %
                               (qarg0, qarg1))

    @staticmethod
    def _broadcast_3_or_more_args(qargs: List) -> List:
        if all(len(qarg) == len(qargs[0]) for qarg in qargs):
            for arg in zip(*qargs):
                yield list(arg), []
        else:
            raise CircuitError(
                'Not sure how to combine these qubit arguments:\n %s\n' % qargs)

    def broadcast_arguments(self, qargs: List, cargs: List) -> Tuple[List, List]:
        """Validation and handling of the arguments and its relationship.

        For example, ``cx([q[0],q[1]], q[2])`` means ``cx(q[0], q[2]); cx(q[1], q[2])``. This
        method yields the arguments in the right grouping. In the given example::

            in: [[q[0],q[1]], q[2]],[]
            outs: [q[0], q[2]], []
                  [q[1], q[2]], []

        The general broadcasting rules are:

            * If len(qargs) == 1::

                [q[0], q[1]] -> [q[0]],[q[1]]

            * If len(qargs) == 2::

                [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]], [q[1], r[1]]
                [[q[0]], [r[0], r[1]]]       -> [q[0], r[0]], [q[0], r[1]]
                [[q[0], q[1]], [r[0]]]       -> [q[0], r[0]], [q[1], r[0]]

            * If len(qargs) >= 3::

                [q[0], q[1]], [r[0], r[1]],  ...] -> [q[0], r[0], ...], [q[1], r[1], ...]

        Args:
            qargs: List of quantum bit arguments.
            cargs: List of classical bit arguments.

        Returns:
            A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits or cargs:
            raise CircuitError(
                'The amount of qubit/clbit arguments does not match the gate expectation.')

        if any([not qarg for qarg in qargs]):
            raise CircuitError('One or more of the arguments are empty')

        if len(qargs) == 1:
            return Gate._broadcast_single_argument(qargs[0])
        elif len(qargs) == 2:
            return Gate._broadcast_2_arguments(qargs[0], qargs[1])
        elif len(qargs) >= 3:
            return Gate._broadcast_3_or_more_args(qargs)
        else:
            raise CircuitError('This gate cannot handle %i arguments' % len(qargs))
