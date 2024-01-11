# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The NumPy eigensolver algorithm."""

from __future__ import annotations

import warnings
from typing import Callable, Union, List, Optional
import logging
import numpy as np
from scipy import sparse as scisparse

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils.validation import validate_min

from .eigensolver import Eigensolver, EigensolverResult
from ..exceptions import AlgorithmError
from ..list_or_dict import ListOrDict

logger = logging.getLogger(__name__)

FilterType = Callable[[Union[List, np.ndarray], float, Optional[ListOrDict[float]]], bool]


class NumPyEigensolver(Eigensolver):
    r"""
    The NumPy eigensolver algorithm.

    The NumPy Eigensolver computes up to the first :math:`k` eigenvalues of a complex-valued square
    matrix of dimension :math:`n \times n`, with :math:`k \leq n`.

    Note:
        Operators are automatically converted to SciPy's ``spmatrix``
        as needed and this conversion can be costly in terms of memory and performance as the
        operator size, mostly in terms of number of qubits it represents, gets larger.
    """

    def __init__(
        self,
        k: int = 1,
        filter_criterion: FilterType | None = None,
    ) -> None:
        """
        Args:
            k: Number of eigenvalues are to be computed, with a minimum value of 1.
            filter_criterion: Callable that allows to filter eigenvalues/eigenstates. Only feasible
                eigenstates are returned in the results. The callable has the signature
                ``filter(eigenstate, eigenvalue, aux_values)`` and must return a boolean to indicate
                whether to keep this value in the final returned result or not. If the number of
                elements that satisfies the criterion is smaller than ``k``, then the returned list will
                have fewer elements and can even be empty.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            validate_min("k", k, 1)

        super().__init__()

        self._in_k = k
        self._k = k

        self._filter_criterion = filter_criterion

    @property
    def k(self) -> int:
        """Return k (number of eigenvalues requested)."""
        return self._in_k

    @k.setter
    def k(self, k: int) -> None:
        """Set k (number of eigenvalues requested)."""
        validate_min("k", k, 1)
        self._in_k = k
        self._k = k

    @property
    def filter_criterion(
        self,
    ) -> FilterType | None:
        """Return the filter criterion if set."""
        return self._filter_criterion

    @filter_criterion.setter
    def filter_criterion(self, filter_criterion: FilterType | None) -> None:
        """Set the filter criterion."""
        self._filter_criterion = filter_criterion

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _check_set_k(self, operator: BaseOperator | PauliSumOp) -> None:
        if operator is not None:
            if self._in_k > 2**operator.num_qubits:
                self._k = 2**operator.num_qubits
                logger.debug(
                    "WARNING: Asked for %s eigenvalues but max possible is %s.", self._in_k, self._k
                )
            else:
                self._k = self._in_k

    def _solve(self, operator: BaseOperator | PauliSumOp) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(operator, PauliSumOp):
            op_matrix = operator.to_spmatrix()
        else:
            try:
                op_matrix = operator.to_matrix(sparse=True)
            except TypeError:
                logger.debug(
                    "WARNING: operator of type `%s` does not support sparse matrices. "
                    "Trying dense computation",
                    type(operator),
                )
                try:
                    op_matrix = operator.to_matrix()
                except AttributeError as ex:
                    raise AlgorithmError(f"Unsupported operator type `{type(operator)}`.") from ex

        if isinstance(op_matrix, scisparse.csr_matrix):
            # If matrix is diagonal, the elements on the diagonal are the eigenvalues. Solve by sorting.
            if scisparse.csr_matrix(op_matrix.diagonal()).nnz == op_matrix.nnz:
                diag = op_matrix.diagonal()
                indices = np.argsort(diag)[: self._k]
                eigval = diag[indices]
                eigvec = np.zeros((op_matrix.shape[0], self._k))
                for i, idx in enumerate(indices):
                    eigvec[idx, i] = 1.0
            else:
                if self._k >= 2**operator.num_qubits - 1:
                    logger.debug(
                        "SciPy doesn't support to get all eigenvalues, using NumPy instead."
                    )
                    eigval, eigvec = self._solve_dense(operator.to_matrix())
                else:
                    eigval, eigvec = self._solve_sparse(op_matrix, self._k)
        else:
            # Sparse SciPy matrix not supported, use dense NumPy computation.
            eigval, eigvec = self._solve_dense(operator.to_matrix())

        indices = np.argsort(eigval)[: self._k]
        eigval = eigval[indices]
        eigvec = eigvec[:, indices]
        return eigval, eigvec.T

    @staticmethod
    def _solve_sparse(op_matrix: scisparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
        if (op_matrix != op_matrix.H).nnz == 0:
            # Operator is Hermitian
            return scisparse.linalg.eigsh(op_matrix, k=k, which="SA")
        else:
            return scisparse.linalg.eigs(op_matrix, k=k, which="SR")

    @staticmethod
    def _solve_dense(op_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if op_matrix.all() == op_matrix.conj().T.all():
            # Operator is Hermitian
            return np.linalg.eigh(op_matrix)
        else:
            return np.linalg.eig(op_matrix)

    @staticmethod
    def _eval_aux_operators(
        aux_operators: ListOrDict[BaseOperator | PauliSumOp],
        wavefn: np.ndarray,
        threshold: float = 1e-12,
    ) -> ListOrDict[tuple[complex, complex]]:

        values: ListOrDict[tuple[complex, complex]]

        # As a list, aux_operators can contain None operators for which None values are returned.
        # As a dict, the None operators in aux_operators have been dropped in compute_eigenvalues.
        if isinstance(aux_operators, list):
            values = [None] * len(aux_operators)
            key_op_iterator = enumerate(aux_operators)
        else:
            values = {}
            key_op_iterator = aux_operators.items()

        for key, operator in key_op_iterator:
            if operator is None:
                continue

            if operator.num_qubits is None or operator.num_qubits < 1:
                logger.info(
                    "The number of qubits of the %s operator must be greater than zero.", key
                )
                continue

            op_matrix = None
            if isinstance(operator, PauliSumOp):
                if operator.coeff != 0:
                    op_matrix = operator.to_spmatrix()
            else:
                try:
                    op_matrix = operator.to_matrix(sparse=True)
                except TypeError:
                    logger.debug(
                        "WARNING: operator of type `%s` does not support sparse matrices. "
                        "Trying dense computation",
                        type(operator),
                    )
                try:
                    op_matrix = operator.to_matrix()
                except AttributeError as ex:
                    raise AlgorithmError(f"Unsupported operator type {type(operator)}.") from ex

            if isinstance(op_matrix, scisparse.csr_matrix):
                value = op_matrix.dot(wavefn).dot(np.conj(wavefn))
            elif isinstance(op_matrix, np.ndarray):
                value = Statevector(wavefn).expectation_value(operator)
            else:
                value = 0.0

            value = value if np.abs(value) > threshold else 0.0
            # The value gets wrapped into a tuple: (mean, metadata).
            # The metadata includes variance (and, for other eigensolvers, shots).
            # Since this is an exact computation, there are no shots
            # and the variance is known to be zero.
            values[key] = (value, {"variance": 0.0})
        return values

    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> NumPyEigensolverResult:

        super().compute_eigenvalues(operator, aux_operators)

        if operator.num_qubits is None or operator.num_qubits < 1:
            raise AlgorithmError("The number of qubits of the operator must be greater than zero.")

        self._check_set_k(operator)

        zero_op = SparsePauliOp(["I" * operator.num_qubits], coeffs=[0.0])
        if isinstance(aux_operators, list) and len(aux_operators) > 0:
            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in aux_operators]
        elif isinstance(aux_operators, dict) and len(aux_operators) > 0:
            aux_operators = {
                key: zero_op if op == 0 else op  # Convert zero values to zero operators
                for key, op in aux_operators.items()
                if op is not None  # Discard None values
            }
        else:
            aux_operators = None

        k_orig = self._k
        if self._filter_criterion:
            # need to consider all elements if a filter is set
            self._k = 2**operator.num_qubits

        eigvals, eigvecs = self._solve(operator)

        # compute energies before filtering, as this also evaluates the aux operators
        if aux_operators is not None:
            aux_op_vals = [
                self._eval_aux_operators(aux_operators, eigvecs[i]) for i in range(self._k)
            ]
        else:
            aux_op_vals = None

        # if a filter is set, loop over the given values and only keep
        if self._filter_criterion:
            filt_eigvals = []
            filt_eigvecs = []
            filt_aux_op_vals = []
            count = 0
            for i, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs)):
                if aux_op_vals is not None:
                    aux_op_val = aux_op_vals[i]
                else:
                    aux_op_val = None

                if self._filter_criterion(eigvec, eigval, aux_op_val):
                    count += 1
                    filt_eigvecs.append(eigvec)
                    filt_eigvals.append(eigval)
                    if aux_op_vals is not None:
                        filt_aux_op_vals.append(aux_op_val)

                if count == k_orig:
                    break

            eigvals = np.array(filt_eigvals)
            eigvecs = np.array(filt_eigvecs)
            aux_op_vals = filt_aux_op_vals

            self._k = k_orig

        result = NumPyEigensolverResult()
        result.eigenvalues = eigvals
        result.eigenstates = [Statevector(vec) for vec in eigvecs]
        result.aux_operators_evaluated = aux_op_vals

        logger.debug("NumpyEigensolverResult:\n%s", result)
        return result


class NumPyEigensolverResult(EigensolverResult):
    """NumPy eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._eigenstates: list[Statevector] | None = None

    @property
    def eigenstates(self) -> list[Statevector] | None:
        """Return eigenstates."""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: list[Statevector]) -> None:
        """Set eigenstates."""
        self._eigenstates = value
