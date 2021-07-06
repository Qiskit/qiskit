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

import warnings
from qiskit.circuit import QuantumCircuit


class UniformDistribution(QuantumCircuit):
    r"""A circuit to encode a discretized uniform distribution in qubit amplitudes.

    This simply corresponds to applying Hadamard gates on all qubits.

    The probability density function of the discretized uniform distribution on
    :math:`N` values is

    .. math::

        \mathbb{P}(X = x) = \frac{1}{N}.

    This circuit considers :math:`N = 2^n`, where :math:`n =` ``num_qubits`` and prepares the state

    .. math::

        \mathcal{P}_X |0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n - 1} |x\rangle

    Examples:
        >>> circuit = UniformDistribution(3)
        >>> circuit.decompose().draw()
             ┌───┐
        q_0: ┤ H ├
             ├───┤
        q_1: ┤ H ├
             ├───┤
        q_2: ┤ H ├
             └───┘

    """

    def __init__(self, num_qubits: int, name: str = "P(X)") -> None:
        """
        Args:
            num_qubits: The number of qubits in the circuit, the distribution is uniform over
                ``2 ** num_qubits`` values.
            name: The name of the circuit.
        """
        warnings.warn(
            "`UniformDistribution` is deprecated as of version 0.17.0 and will be "
            "removed no earlier than 3 months after the release date. "
            "It moved to qiskit_finance.circuit.library.UniformDistribution.",
            DeprecationWarning,
            stacklevel=2,
        )

        circuit = QuantumCircuit(num_qubits, name=name)
        circuit.h(circuit.qubits)

        super().__init__(*circuit.qregs, name=name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
