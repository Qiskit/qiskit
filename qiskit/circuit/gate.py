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

    def __init__(self, name, num_qubits, params):
        """Create a new composite gate.

        name = gate name string
        num_qubits = number of qubits the gate acts on
        params = list of parameters
        """
        super().__init__(name, num_qubits, 0, params)

    def to_matrix(self):
        """Return a Numpy.array for the gate unitary matrix.

        Additional Information
        ----------------------
        If a Gate subclass does not implement this method an exception
        will be raised when this base class method is called.
        """
        raise QiskitError("to_matrix not defined for this {}".format(type(self)))
