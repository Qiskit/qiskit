# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Base class for readout error mitigation.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional, List, Iterable, Tuple, Union, Callable
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.result import Counts, QuasiDistribution

logger = logging.getLogger(__name__)


class BaseReadoutMitigator(ABC):
    """Base readout error mitigator class."""

    @abstractmethod
    def quasi_probabilities(
        self,
        data: Counts,
        qubits: Iterable[int] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> (QuasiDistribution, QuasiDistribution):
        """Convert counts to a dictionary of quasi-probabilities

        Args:
            data: Counts to be mitigated.
            qubits: the physical qubits measured to obtain the counts clbits.
                If None these are assumed to be qubits [0, ..., N-1]
                for N-bit counts.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            QuasiDistibution: A dictionary containing pairs of [output, mean] where "output"
                is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state,
                and "mean" is the mean of non-zero quasi-probability estimates.
            QuasiDistibution: A dictionary containing pairs of [output, standard deviation]
                where "output" is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state,
                and "standard deviation" is the standard deviation of the non-zero
                quasi-probability estimates.
        """

    @abstractmethod
    def expectation_value(
        self,
        data: Counts,
        diagonal: Union[Callable, dict, str, np.ndarray],
        qubits: Iterable[int] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Calculate the expectation value of a diagonal Hermitian operator.

        Args:
            data: Counts object to be mitigated.
            diagonal: the diagonal operator. This may either be specified
                      as a string containing I,Z,0,1 characters, or as a
                      real valued 1D array_like object supplying the full diagonal,
                      or as a dictionary, or as Callable.
            qubits: the physical qubits measured to obtain the counts clbits.
                    If None these are assumed to be qubits [0, ..., N-1]
                    for N-bit counts.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            The mean and standard deviation of operator expectation value
            calculated from the current counts.
        """

    @staticmethod
    def _z_diagonal(dim, dtype=float):
        r"""Return the diagonal for the operator :math:`Z^\otimes n`"""
        parity = np.zeros(dim, dtype=dtype)
        for i in range(dim):
            parity[i] = bin(i)[2:].count("1")
        return (-1) ** np.mod(parity, 2)

    @staticmethod
    def _expval_with_stddev(
        coeffs: np.ndarray, probs: np.ndarray, shots: int
    ) -> Tuple[float, float]:
        """Compute expectation value and standard deviation.
        Args:
            coeffs: array of diagonal operator coefficients.
            probs: array of measurement probabilities.
            shots: total number of shots to obtain probabilities.
        Returns:
            tuple: (expval, stddev) expectation value and standard deviation.
        """
        # Compute expval
        expval = coeffs.dot(probs)

        # Compute variance
        sq_expval = (coeffs ** 2).dot(probs)
        variance = (sq_expval - expval ** 2) / shots

        # Compute standard deviation
        if variance < 0 and not np.isclose(variance, 0):
            logger.warning(
                "Encountered a negative variance in expectation value calculation."
                "(%f). Setting standard deviation of result to 0.",
                variance,
            )
        stddev = np.sqrt(variance) if variance > 0 else 0.0
        return [expval, stddev]

    @staticmethod
    def _stddev(probs, shots):
        """Calculate stddev dict"""
        ret = {}
        for key, prob in probs.items():
            std_err = np.sqrt(prob * (1 - prob) / shots)
            ret[key] = std_err
        return ret

    @staticmethod
    def _str2diag(string):
        chars = {
            "I": np.array([1, 1], dtype=float),
            "Z": np.array([1, -1], dtype=float),
            "0": np.array([1, 0], dtype=float),
            "1": np.array([0, 1], dtype=float),
        }
        ret = np.array([1], dtype=float)
        for i in string:
            if i not in chars:
                raise QiskitError(f"Invalid diagonal string character {i}")
            ret = np.kron(chars[i], ret)
        return ret

    def _stddev_upper_bound(self, shots, qubits):
        """Return an upper bound on standard deviation of expval estimator.
        Args:
            shots: Number of shots used for expectation value measurement.
            qubits: qubits being measured for operator expval.
        Returns:
            float: the standard deviation upper bound.
        """
        gamma = self._compute_gamma(qubits=qubits)
        return gamma / np.sqrt(shots)

    @abstractmethod
    def _compute_gamma(self, qubits=None) -> float:
        """Compute gamma for N-qubit mitigation"""
