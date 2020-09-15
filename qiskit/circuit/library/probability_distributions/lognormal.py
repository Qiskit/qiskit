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

"""The log-normal probability distribution circuit."""

from typing import Tuple, List, Union, Optional
import numpy as np
from scipy.stats import multivariate_normal
from qiskit.circuit import QuantumCircuit
from .normal import _check_bounds_valid, _check_dimensions_match


class LogNormalDistribution(QuantumCircuit):
    r"""A circuit to encode a discretized log-normal distribution in qubit amplitudes.

    A random variable :math:`X` is log-normal distributed if

    .. math::

        \log(X) \sim \mathcal{N}(\mu, \sigma^2)

    for a normal distribution :math:`\mathcal{N}(\mu, \sigma^2)`.
    The probability density function of the log-normal distribution is defined as

    .. math::

        \mathbb{P}(X = x) = \frac{1}{x\sqrt{2\pi\sigma^2}} e^{-\frac{(\log(x) - \mu)^2}{\sigma^2}}


    This circuit considers the discretized version of :math:`X` on ``2 ** num_qubits`` equidistant
    points, :math:`x_i`, truncated to ``bounds``. See also
    :class:`~qiskit.circuit.library.NormalDistribution` for more information.

    This circuit is for example used in amplitude estimation applications, such as finance [1, 2],
    where customer demand or the return of a portfolio could be modelled using a log-normal
    distribution.

    Examples:
        This class can be used for both univariate and multivariate distributions.
        >>> mu = [1, 0.9, 0.2]
        >>> sigma = [[1, -0.2, 0.2], [-0.2, 1, 0.4], [0.2, 0.4, 1]]
        >>> circuit = LogNormalDistribution([2, 2, 2], mu, sigma)
        >>> circuit.num_qubits
        6

    References:
        [1]: Gacon, J., Zoufal, C., & Woerner, S. (2020).
             Quantum-Enhanced Simulation-Based Optimization.
             `arXiv:2005.10780 <http://arxiv.org/abs/2005.10780>`_

        [2]: Woerner, S., & Egger, D. J. (2018).
             Quantum Risk Analysis.
             `arXiv:1806.06893 <http://arxiv.org/abs/1806.06893>`_

    """

    def __init__(self,
                 num_qubits: Union[int, List[int]],
                 mu: Union[float, List[float]] = 0,
                 sigma: Union[float, List[float]] = 1,
                 bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                 name: str = 'P(X)') -> None:
        r"""
        Args:
            num_qubits: The number of qubits used to discretize the random variable. For a 1d
                random variable, ``num_qubits`` is an integer, for multiple dimensions a list
                of integers indicating the number of qubits to use in each dimension.
            mu: The parameter :math:`\mu` of the distribution.
                Can be either a float for a 1d random variable or a list of floats for a higher
                dimensional random variable.
            sigma: The parameter :math:`\sigma`, which is the standard deviation or covariance
                matrix.
            bounds: The truncation bounds of the distribution as tuples. For multiple dimensions,
                ``bounds`` is a list of tuples ``[(low0, high0), (low1, high1), ...]``.
                If ``None``, the bounds are set to ``(0, 1)`` for each dimension.
            name: The name of the circuit.
        """
        _check_dimensions_match(num_qubits, mu, sigma, bounds)
        _check_bounds_valid(bounds)

        # set default arguments
        dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
        if mu is None:
            mu = 0 if dim == 1 else [0] * dim

        if sigma is None:
            sigma = 1 if dim == 1 else np.eye(dim)

        if bounds is None:
            bounds = (0, 1) if dim == 1 else [(0, 1)] * dim

        if not isinstance(num_qubits, list):  # univariate case
            super().__init__(num_qubits, name=name)

            x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)  # evaluation points
        else:  # multivariate case
            super().__init__(sum(num_qubits), name=name)

            # compute the evaluation points using numpy's meshgrid
            # indexing 'ij' yields the "column-based" indexing
            meshgrid = np.meshgrid(*[np.linspace(bound[0], bound[1], num=2**num_qubits[i])
                                     for i, bound in enumerate(bounds)], indexing='ij')
            # flatten into a list of points
            x = list(zip(*[grid.flatten() for grid in meshgrid]))

        # compute the normalized, truncated probabilities
        probabilities = []
        for x_i in x:
            # pylint: disable=line-too-long
            # map probabilities from normal to log-normal reference:
            # https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
            if np.min(x_i) > 0:
                det = 1 / np.prod(x_i)
                probability = multivariate_normal.pdf(np.log(x_i), mu, sigma) * det
            else:
                probability = 0
            probabilities += [probability]
        normalized_probabilities = probabilities / np.sum(probabilities)

        # use default synthesis to construct the circuit
        self.initialize(np.sqrt(normalized_probabilities), self.qubits)  # pylint: disable=no-member
