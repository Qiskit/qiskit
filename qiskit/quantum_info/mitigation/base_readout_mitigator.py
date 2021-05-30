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
from typing import Optional, List, Dict, Iterable, Tuple, Union
import numpy as np
from qiskit.result import Counts

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class BaseReadoutMitigator(ABC):
    """Base readout error mitigator class."""

    def quasi_probabilities(data: Counts,
                            qubits: Iterable[int] = None,
                            shots: Optional[int] = None
                            ) -> (Dict[str, float], Dict[str, float]):
        """Convert counts to a dictionary of non-zero probabilities

        Args:
            data: Counts to be mitigated.
            qubits: the physical qubits measured to obtain the counts clbits.
                    If None these are assumed to be qubits [0, ..., N-1]
                    for N-bit counts.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            A dictionary containing pairs of [output, mean] where "output" is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state, and "mean" is the mean
                of non-zero quasi-probability estimates.
            A dictionary containing pairs of [output, standard deviation] where "output" is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state, and "standard deviation" is the
                standard deviation of the non-zero quasi-probability estimates.

        Raises:
            QiskitError: if qubits is not None and does not match the number of count clbits.
        """

    def expectation_value(data: Counts,
                          diagonal: np.ndarray,
                          qubits: Iterable[int] = None,
                          shots: Optional[int] = None
                          ) -> List[float, float]:
        """Calculate the expectation value of a diagonal Hermitian operator.

        Args:
            data: Counts object to be mitigated.
            diagonal: the diagonal operator. This may either be specified
                      as a string containing I,Z,0,1 characters, or as a
                      real valued 1D array_like object.
            qubits: the physical qubits measured to obtain the counts clbits.
                    If None these are assumed to be qubits [0, ..., N-1]
                    for N-bit counts.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            The mean and standard deviation of operator expectation value
            calculated from the current counts.

        Raises:
            QiskitError: if the diagonal does not match the number of count clbits.
            QiskitError: if qubits is not None and does not match the number of count clbits.
        """

    @abstractmethod
    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement mitigation matrix for the specified qubits.
        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """

    @abstractmethod
    def assignment_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement assignment matrix for specified qubits.
        The assignment matrix is the stochastic matrix :math:`A` which assigns
        a noisy measurement probability distribution to an ideal input
        measurement distribution: :math:`P(i|j) = \langle i | A | j \rangle`.
        Args:
            qubits: Optional, qubits being measured for operator expval.
        Returns:
            np.ndarray: the assignment matrix A.
        """

    def assignment_fidelity(self, qubits: Optional[List[int]] = None) -> float:
        r"""Return the measurement assignment fidelity on the specified qubits.
        The assignment fidelity on N-qubits is defined as
        :math:`\sum_{x\in\{0, 1\}^n} P(x|x) / 2^n`, where
        :math:`P(x|x) = \rangle x|A|x\langle`, and :math:`A` is the
        :meth:`assignment_matrix`.

        Args:
            qubits: Optional, qubits being measured for operator expval.

        Returns:
            float: the assignment fidelity.
        """
        return self.assignment_matrix(qubits=qubits).diagonal().mean()


    def plot_assignment_matrix(self,
                               qubits=None,
                               ax=None):
        """Matrix plot of the readout error assignment matrix.

        Args:
            qubits (list(int)): Optional, qubits being measured for operator expval.
            ax (axes): Optional. Axes object to add plot to.

        Returns:
            plt.axes: the figure axes object.

        Raises:
            ImportError: if matplotlib is not installed.
        """

    def plot_mitigation_matrix(self,
                               qubits=None,
                               ax=None):
        """Matrix plot of the readout error mitigation matrix.

            Args:
                qubits (list(int)): Optional, qubits being measured for operator expval.
                ax (plt.axes): Optional. Axes object to add plot to.

            Returns:
                plt.axes: the figure axes object.

            Raises:
                ImportError: if matplotlib is not installed.
        """
