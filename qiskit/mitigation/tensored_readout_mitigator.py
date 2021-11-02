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
Readout mitigator class based on the 1-qubit tensored mitigation method
"""


from typing import Optional, List, Tuple, Iterable, Callable, Union
import numpy as np

from qiskit.result import Counts, QuasiDistribution
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, stddev, expval_with_stddev, z_diagonal, str2diag


class TensoredReadoutMitigator(BaseReadoutMitigator):
    """1-qubit tensor product readout error mitigator.
    Mitigates expectation_value and quasi_probabilities.
    The mitigator should either be calibrated using qiskit.experiments,
    or calculated directly from the backend properties."""

    def __init__(self, amats: List[np.ndarray] = None, backend: str = None):
        """Initialize a TensoredReadoutMitigator
        Args:
            amats: Optional, list of single-qubit readout error assignment matrices.
            backend: Optional, backend name.
        """
        if amats is None:
            amats = self._from_backend(backend)
        self._num_qubits = len(amats)
        self._assignment_mats = amats
        self._mitigation_mats = np.zeros([self._num_qubits, 2, 2], dtype=float)
        self._gammas = np.zeros(self._num_qubits, dtype=float)

        for i in range(self._num_qubits):
            mat = self._assignment_mats[i]
            # Compute Gamma values
            error0 = mat[1, 0]
            error1 = mat[0, 1]
            self._gammas[i] = (1 + abs(error0 - error1)) / (1 - error0 - error1)
            # Compute inverse mitigation matrix
            try:
                ainv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                ainv = np.linalg.pinv(mat)
            self._mitigation_mats[i] = ainv

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
        probs_vec, shots = counts_probability_vector(
            data, clbits=clbits, qubits=qubits, return_shots=True
        )

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = range(self._num_qubits)
        ainvs = self._mitigation_mats[list(qubits)]

        # Get operator coeffs
        if diagonal is None:
            diagonal = z_diagonal(2 ** self._num_qubits)
        elif isinstance(diagonal, str):
            diagonal = str2diag(diagonal)

        # Apply transpose of mitigation matrix
        coeffs = np.reshape(diagonal, self._num_qubits * [2])
        einsum_args = [coeffs, list(range(self._num_qubits))]
        for i, ainv in enumerate(reversed(ainvs)):
            einsum_args += [ainv.T, [self._num_qubits + i, i]]
        einsum_args += [list(range(self._num_qubits, 2 * self._num_qubits))]
        coeffs = np.einsum(*einsum_args).ravel()

        return expval_with_stddev(coeffs, probs_vec, shots)

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
        probs_vec, shots = counts_probability_vector(
            data, clbits=clbits, qubits=qubits, return_shots=True
        )

        # Get qubit mitigation matrix and mitigate probs
        if qubits is None:
            qubits = range(self._num_qubits)
        mit_mat = self.mitigation_matrix(qubits)

        # Apply transpose of mitigation matrix
        probs_vec = mit_mat.dot(probs_vec)
        probs_dict = {}
        for index, _ in enumerate(probs_vec):
            probs_dict[index] = probs_vec[index]

        return QuasiDistribution(probs_dict), QuasiDistribution(stddev(
            QuasiDistribution(probs_dict).nearest_probability_distribution(), shots)
        )

    def mitigation_matrix(self, qubits: List[int] = None) -> np.ndarray:
        r"""Return the measurement mitigation matrix for the specified qubits.
        The mitigation matrix :math:`A^{-1}` is defined as the inverse of the
        :meth:`assignment_matrix` :math:`A`.
        Args:
            qubits: Optional, qubits being measured for operator expval.
        Returns:
            np.ndarray: the measurement error mitigation matrix :math:`A^{-1}`.
        """
        if qubits is None:
            qubits = list(range(self._num_qubits))
        if isinstance(qubits, int):
            qubits = [qubits]
        mat = self._mitigation_mats[qubits[0]]
        for i in qubits[1:]:
            mat = np.kron(self._mitigation_mats[i], mat)
        return mat

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
        if qubits is None:
            qubits = list(range(self._num_qubits))
        if isinstance(qubits, int):
            qubits = [qubits]
        mat = self._assignment_mats[qubits[0]]
        for i in qubits[1:]:
            mat = np.kron(self._assignment_mats[qubits[i]], mat)
        return mat

    def _compute_gamma(self, qubits=None):
        """Compute gamma for N-qubit mitigation"""
        if qubits is None:
            gammas = self._gammas
        else:
            gammas = self._gammas[list(qubits)]
        return np.product(gammas)

    def _from_backend(self, backend: str):
        """Calculates amats from backend properties readout_error"""
        num_qubits = len(backend.properties().qubits)
        amats = np.zeros([num_qubits, 2, 2], dtype=float)

        for qubit_idx, qubit_prop in enumerate(backend.properties().qubits):
            for prop in qubit_prop:
                if prop.name == "prob_meas0_prep1":
                    (amats[qubit_idx])[0, 1] = prop.value
                    (amats[qubit_idx])[1, 1] = 1 - prop.value
                if prop.name == "prob_meas1_prep0":
                    (amats[qubit_idx])[1, 0] = prop.value
                    (amats[qubit_idx])[0, 0] = 1 - prop.value

        return amats
