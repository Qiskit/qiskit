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
Readout mitigator class based on the A-matrix inversion method
"""

import logging
from typing import Optional, List, Tuple, Iterable, Callable, Union
import numpy as np

from qiskit.result import Counts, marginal_counts, QuasiDistribution
from .base_readout_mitigator import BaseReadoutMitigator

logger = logging.getLogger(__name__)


class CompleteReadoutMitigator(BaseReadoutMitigator):
    """N-qubit readout error mitigator.
    Mitigates expectation_value and quasi_probabilities.
    The mitigation_matrix should be calibrated using qiskit.experiments."""

    def __init__(self, amat: np.ndarray):
        """Initialize a CompleteReadoutMitigator
        Args:
            amat (np.array): readout error assignment matrix.
        """
        self._num_qubits = int(np.log2(amat.shape[0]))
        self._assignment_mat = amat
        self._mitigation_mats = {}

    def expectation_value(
        self,
        data: Counts,
        diagonal: Union[Callable, dict, str, np.ndarray] = None,
        qubits: Iterable[int] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[int] = None,
    ) -> Tuple[float, float]:
        r"""Compute the mitigated expectation value of a diagonal observable.
        This computes the mitigated estimator of
        :math:`\langle O \rangle = \mbox{Tr}[\rho. O]` of a diagonal observable
        :math:`O = \sum_{x\in\{0, 1\}^n} O(x)|x\rangle\!\langle x|`.

        Args:
            data: Counts object
            diagonal: Optional, the vector of diagonal values for summing the
                      expectation value. If ``None`` the the default value is
                      :math:`[1, -1]^\otimes n`.
            qubits: Optional, the measured physical qubits the count
                    bitstrings correspond to. If None qubits are assumed to be
                    :math:`[0, ..., n-1]`.
            clbits: Optional, if not None marginalize counts to the specified bits.
            shots: the number of shots.

        Returns:
            (float, float): the expectation value and standard deviation.

        Raises:
            QiskitError: if input arguments are invalid.

        Additional Information:
            The diagonal observable :math:`O` is input using the ``diagonal`` kwarg as
            a list or Numpy array :math:`[O(0), ..., O(2^n -1)]`. If no diagonal is specified
            the diagonal of the Pauli operator
            :math`O = \mbox{diag}(Z^{\otimes n}) = [1, -1]^{\otimes n}` is used.
            The ``clbits`` kwarg is used to marginalize the input counts dictionary
            over the specified bit-values, and the ``qubits`` kwarg is used to specify
            which physical qubits these bit-values correspond to as
            ``circuit.measure(qubits, clbits)``.
        """
        # Marginalize counts
        if clbits is not None:
            data = marginal_counts(data, clbits)

        # Get probability vector
        num_qubits = data.size()
        probs_vec = self._to_probs_vec(data, num_qubits)
        shots = data.shots()

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = range(num_qubits)
        mit_mat = self.mitigation_matrix(qubits)

        # Get operator coeffs
        if diagonal is None:
            diagonal = self._z_diagonal(2 ** num_qubits)
        else:
            diagonal = self._str2diag(diagonal)

        # Apply transpose of mitigation matrix
        coeffs = mit_mat.T.dot(diagonal)

        return self._expval_with_stddev(coeffs, probs_vec, shots)

    def quasi_probabilities(
        self,
        data: Counts,
        qubits: Optional[List[int]] = None,
        clbits: Optional[List[int]] = None,
        shots: Optional[bool] = False,
    ) -> (QuasiDistribution, QuasiDistribution):
        """Compute mitigated quasi probabilities value.

        Args:
            data: counts object
            qubits: qubits the count bitstrings correspond to.
            clbits: Optional, marginalize counts to just these bits.
            shots: the number of shots.

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

        Raises:
            QiskitError: if qubit and clbit kwargs are not valid.
        """
        # Marginalize counts
        if clbits is not None:
            data = marginal_counts(data, clbits)

        # Get total number of qubits and shots
        num_qubits = data.size()
        shots = data.shots()
        # Get probability vector
        probs_vec = self._to_probs_vec(data, num_qubits)

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = range(num_qubits)
        mit_mat = self.mitigation_matrix(qubits)

        # Apply transpose of mitigation matrix
        probs_vec = mit_mat.dot(probs_vec)
        probs_dict = {}
        for index, _ in enumerate(probs_vec):
            probs_dict[index] = probs_vec[index]

        return QuasiDistribution(probs_dict), QuasiDistribution(self._stddev(probs_dict, shots))

    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the readout mitigation matrix for the specified qubits.
        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.
        Args:
            qubits: Optional, qubits being measured.
        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
        if qubits is None:
            qubits = tuple(range(self._num_qubits))
        else:
            qubits = tuple(sorted(qubits))

        # Check for cached mitigation matrix
        # if not present compute
        if qubits not in self._mitigation_mats:
            marginal_matrix = self.assignment_matrix(qubits)
            try:
                mit_mat = np.linalg.inv(marginal_matrix)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if matrix is singular
                mit_mat = np.linalg.pinv(marginal_matrix)
            self._mitigation_mats[qubits] = mit_mat

        return self._mitigation_mats[qubits]

    def assignment_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the readout assignment matrix for specified qubits.
        The assignment matrix is the stochastic matrix :math:`A` which assigns
        a noisy readout probability distribution to an ideal input
        readout distribution: :math:`P(i|j) = \langle i | A | j \rangle`.
        Args:
            qubits: Optional, qubits being measured.
        Returns:
            np.ndarray: the assignment matrix A.
        """
        if qubits is None:
            return self._assignment_mat

        if isinstance(qubits, int):
            qubits = [qubits]

        # Compute marginal matrix
        axis = tuple(
            self._num_qubits - 1 - i for i in set(range(self._num_qubits)).difference(qubits)
        )
        num_qubits = len(qubits)
        new_amat = np.zeros(2 * [2 ** num_qubits], dtype=float)
        for i, col in enumerate(self._assignment_mat.T[self._keep_indexes(qubits)]):
            new_amat[i] = (
                np.reshape(col, self._num_qubits * [2]).sum(axis=axis).reshape([2 ** num_qubits])
            )
        new_amat = new_amat.T
        return new_amat

    @staticmethod
    def _keep_indexes(qubits):
        indexes = [0]
        for i in sorted(qubits):
            indexes += [idx + (1 << i) for idx in indexes]
        return indexes

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        mitmat = self.mitigation_matrix(qubits=qubits)
        return np.max(np.sum(np.abs(mitmat), axis=0))
