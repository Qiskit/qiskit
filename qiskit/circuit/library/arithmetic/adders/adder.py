# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two equally sized qubit registers."""

from qiskit.circuit import QuantumCircuit


class Adder(QuantumCircuit):
    r"""Compute the sum of two equally sized qubit registers.

    For two registers :math:`|a\rangle_n` and :math:|b\rangle_n` with :math:`n` qubits each, an
    adder performs the following operation

    .. math::

        |a\rangle_n |b\rangle_n \mapsto |a\rangle_n |a + b\rangle_{n + 1}.

    The quantum register :math:`|a\rangle_n` (and analogously :math:`|b\rangle_n`)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(self, num_state_qubits: int, name: str = "Adder") -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the registers.
            name: The name of the circuit.
        """
        super().__init__(name=name)
        self._num_state_qubits = num_state_qubits

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits, i.e. the number of bits in each input register.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits
