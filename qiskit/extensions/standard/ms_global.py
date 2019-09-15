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

# pylint: disable=invalid-name

"""
Molmer-Sorensen gate.
"""

import numpy

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cu1 import Cu1Gate
from qiskit.extensions.standard.ms import MSGate


class MSGlobalGate(Gate):
    """ Global Molmer-Sorensen gate."""

    def __init__(self, N_qubits, theta):
        """Create new MS gate."""
        super().__init__("ms_global", N_qubits, [N_qubits, theta])

    def _define(self):
        definition = []
        q = QuantumRegister(self.params[0], "q")
        rule = []
        for i in range(self.num_qubits):
            for j in range(i, self.num_qubits):
                rule += [(MSGate(self.params[0]), [q[i], q[j]], [])]

        for inst in rule:
            definition.append(inst)
        self.definition = definition

        print(self.num_qubits)
        print(self.definition)



    #def to_matrix(self):
    #    """Return a Numpy.array for the MS gate."""
    #    # ToDo
    #    return numpy.array([[1]])


def ms_global(self, theta,  qubits):
    """Apply MS to q1 and q2."""
    return self.append(MSGlobalGate(len(qubits), theta), qubits)


QuantumCircuit.ms_global = ms_global
