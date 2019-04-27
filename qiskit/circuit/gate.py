# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
