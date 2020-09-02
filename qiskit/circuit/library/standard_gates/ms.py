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

"""Global Mølmer–Sørensen gate."""

from qiskit.util import deprecate_arguments
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class MSGate(Gate):
    """Global Mølmer–Sørensen gate.

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS can be
    applied to multiple ions to entangle multiple qubits simultaneously.

    In the two-qubit case, this is equivalent to an XX(theta) interaction,
    and is thus reduced to the RXXGate.
    """

    @deprecate_arguments({'n_qubits': 'num_qubits'})
    def __init__(self, num_qubits, theta, *, n_qubits=None,  # pylint:disable=unused-argument
                 label=None):
        """Create new MS gate."""
        super().__init__('ms', num_qubits, [theta], label=label)

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .rxx import RXXGate
        theta = self.params[0]
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                qc._append(RXXGate(theta), [q[i], q[j]], [])

        self.definition = qc
