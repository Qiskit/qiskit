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

"""The normal probability distribution circuit."""

import numpy
import scipy
from qiskit.circuit import QuantumCircuit


class NormalDistribution(QuantumCircuit):
    r"""The normal distribution circuit.
    
    The probability density function of the normal distribution is defined as

    .. math::

        \mathbb{P}(X = x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{\sigma^2}}

    """

    def __init__(self, num_qubits: int, mu: float = 0, sigma: float = 1, 
                 bounds: Tuple[float, float] = (0, 1), name: str = 'P(X)') -> None:
        """
        Args:
            num_qubits: The number of qubits in the circuit, the distribution is uniform over 
                ``2 ** num_qubits`` values.
            mu: The parameter :math:`\mu`, which is the expected value of the distribution.
            sigma: The parameter :math:`\sigma`, which is the standard deviation.
            bounds: The truncation bounds of the distribution.
            name: The name of the circuit.
        """
        super().__init__(num_qubits, name=name)
        
        # compute the normalized, truncated probabilities 
        x = numpy.linspace(bounds[0], bounds[1], num=2**num_qubits)
        probabilities = [scipy.stats.norm(x_i, mu, sigma) for x_i in x]
        normalized_probabilites = probabilities / numpy.sum(probabilities)

        # use default synthesis to construct the circuit
        self.initialize(normalized_probabilities)

