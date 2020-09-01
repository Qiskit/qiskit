# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The uniform probability distribution circuit."""

from qiskit.circuit import QuantumCircuit

class UniformDistribution(QuantumCircuit):
    """The uniform distribution circuit."""

    def __init__(self, num_qubits: int, name: str = 'P(X)') -> None:
        """
        Args:
            num_qubits: The number of qubits in the circuit, the distribution is uniform over 
                ``2 ** num_qubits`` values.
        """
        super().__init__(num_qubits, name=name)
        self.h(self.qubits)
