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

from typing import Tuple, Union, List, Optional
import numpy as np
from scipy.stats import multivariate_normal
from qiskit.circuit import QuantumCircuit


class NormalDistribution(QuantumCircuit):
    r"""The normal distribution circuit.

    The probability density function of the normal distribution is defined as

    .. math::

        \mathbb{P}(X = x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{\sigma^2}}

    """

    def __init__(self,
                 num_qubits: Union[int, List[int]],
                 mu: Union[float, List[float]] = 0,
                 sigma: Union[float, List[float]] = 1,
                 bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                 name: str = 'P(X)') -> None:
        r"""
        Args:
            num_qubits: The number of qubits in the circuit, the distribution is uniform over
                ``2 ** num_qubits`` values.
            mu: The parameter :math:`\mu`, which is the expected value of the distribution.
            sigma: The parameter :math:`\sigma`, which is the standard deviation.
            bounds: The truncation bounds of the distribution.
            name: The name of the circuit.
        """
        if not isinstance(num_qubits, list):  # univariate case
            super().__init__(num_qubits, name=name)

            if bounds is None:
                bounds = (-1, 1)

            x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)

        else:  # multivariate case
            super().__init__(sum(num_qubits), name=name)

            if bounds is None:
                bounds = [(-1, 1)] * len(num_qubits)

            # compute the evaluation points using numpy's meshgrid
            # indexing 'ij' yields the "column-based" indexing
            meshgrid = np.meshgrid(*[np.linspace(bound[0], bound[1], num=2**num_qubits[i])
                                     for i, bound in enumerate(bounds)], indexing='ij')
            # flatten into a list of points
            x = list(zip(*[grid.flatten() for grid in meshgrid]))

        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, mu, sigma)
        normalized_probabilities = probabilities / np.sum(probabilities)

        # use default synthesis to construct the circuit
        self.initialize(np.sqrt(normalized_probabilities), self.qubits)  # pylint: disable=no-member
