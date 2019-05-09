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

"""
Unitary gate.
"""

from qiskit.exceptions import QiskitError
from .instruction import Instruction


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, num_qubits, params, label=None):
        """Create a new gate.

        Args:
            name (str): the Qobj name of the gate
            num_qubits (int): the number of qubits the gate acts on.
            params (list): a list of parameters.
            label (str or None): An optional label for the gate [Default: None]
        """
        self._label = label
        super().__init__(name, num_qubits, 0, params)

    def to_matrix(self):
        """Return a Numpy.array for the gate unitary matrix.

        Additional Information
        ----------------------
        If a Gate subclass does not implement this method an exception
        will be raised when this base class method is called.
        """
        raise QiskitError("to_matrix not defined for this {}".format(type(self)))

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        instruction = super().assemble()
        if self.label:
            instruction.label = self.label
        return instruction

    @property
    def label(self):
        """Return gate label"""
        return self._label

    @label.setter
    def label(self, name):
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

    @staticmethod
    def _broadcast_single_argument(qarg):
        """ Expands a single argument. For example: [q[0], q[1]] -> [q[0]], [q[1]]
        """
        # [q[0], q[1]] -> [q[0]]
        #              -> [q[1]]
        for arg0 in qarg:
            yield [arg0], []

    @staticmethod
    def _broadcast_2_arguments(qarg0, qarg1):
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
            raise QiskitError('Not sure how to combine these two qubit arguments:\n %s\n %s' %
                              (qarg0, qarg1))

    @staticmethod
    def _broadcast_3_arguments(qarg0, qarg1, qarg2):
        if len(qarg0) == len(qarg1) == len(qarg2):
            # [q[0], q[1]], [r[0], r[1]],  [s[0], s[1]] -> [q[0], r[0], s[0]]
            #                                           -> [q[1], r[1], s[1]]
            for arg0, arg1, arg2 in zip(qarg0, qarg1, qarg2):
                yield [arg0, arg1, arg2], []
        else:
            raise QiskitError(
                'Not sure how to combine these three qubit arguments:\n %s\n %s\n %s' %
                (qarg0, qarg1, qarg2))

    def broadcast_arguments(self, qargs, cargs):
        """
        Validation and handling of the arguments and its relationship. For example:
        `cx([q[0],q[1]], q[2])` means `cx(q[0], q[2]); cx(q[1], q[2])`. This method
        yields the arguments in the right grouping. In the given example:
          in: [[q[0],q[1]], q[2]],[]
        outs: [q[0], q[2]], []
              [q[1], q[2]], []
        The general broadcasting rules are:
         * If len(qargs) == 1:
                [q[0], q[1]] -> [q[0]],[q[1]]
         * If len(qargs) == 2:
                [[q[0], q[1]], [r[0], r[1]]] -> [q[0], r[0]], [q[1], r[1]]
                [[q[0]], [r[0], r[1]]]       -> [q[0], r[0]], [q[0], r[1]]
                [[q[0], q[1]], [r[0]]]       -> [q[0], r[0]], [q[1], r[0]]
         * If len(qargs) == 3:
                [q[0], q[1]], [r[0], r[1]],  [s[0], s[1]] -> [q[0], r[0], s[0]], [q[1], r[1], s[1]]

        Args:
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Returns:
            Tuple(List, List): A tuple with single arguments.

        Raises:
            QiskitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qargs) != self.num_qubits or cargs:
            raise QiskitError(
                'The amount of qubit/clbit arguments does not match the gate expectation.')

        if any([not qarg for qarg in qargs]):
            raise QiskitError('One or more of the arguments are empty')

        if len(qargs) == 1:
            return Gate._broadcast_single_argument(qargs[0])
        elif len(qargs) == 2:
            return Gate._broadcast_2_arguments(qargs[0], qargs[1])
        elif len(qargs) == 3:
            return Gate._broadcast_3_arguments(qargs[0], qargs[1], qargs[2])
        else:
            raise QiskitError('This gate cannot handle %i arguments' % len(qargs))
