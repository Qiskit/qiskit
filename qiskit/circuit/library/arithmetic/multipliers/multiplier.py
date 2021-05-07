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

"""Compute the product of two equally sized qubit registers."""

from qiskit.circuit import QuantumCircuit


class Multiplier(QuantumCircuit):
    r"""Compute the product of two equally sized qubit registers.

    For two input registers :math:`|a\rangle_n`, :math:`|b\rangle_n` with :math:`n` qubits each
    and an output register with :math:`2n` qubits, a multiplier performs the following operation

    .. math::

        |a\rangle |b\rangle |0\rangle \mapsto |a\rangle |b\rangle |a \cdot b\rangle

    The quantum register :math:`|a\rangle_n` (analogously :math:`|b\rangle_n` and
    output register)

    .. math::

        |a\rangle_n = |a_0\rangle \otimes \cdots \otimes |a_{n - 1}\rangle,

    for :math:`a_i \in \{0, 1\}`, is associated with the integer value

    .. math::

        a = 2^{0}a_{0} + 2^{1}a_{1} + \cdots + 2^{n - 1}a_{n - 1}.

    """

    def __init__(self, num_state_qubits: int, name: str = 'Multiplier') -> None:
        """
        Args:
            num_state_qubits: The number of qubits in each of the input registers.
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
