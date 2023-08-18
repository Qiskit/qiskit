# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Eigensolver algorithm."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
from scipy import sparse as scisparse

from qiskit.opflow import I, ListOp, OperatorBase, StateFn
from qiskit.utils.validation import validate_min
from qiskit.utils.deprecation import deprecate_func
from ..exceptions import AlgorithmError
from .eigen_solver import Eigensolver, EigensolverResult
from ..list_or_dict import ListOrDict

logger = logging.getLogger(__name__)


class NumPyEigensolver(Eigensolver):
    r"""
    Deprecated: NumPy Eigensolver algorithm.

    The NumPyEigensolver class has been superseded by the
    :class:`qiskit.algorithms.eigensolvers.NumPyEigensolver` class.
    This class will be deprecated in a future release and subsequently
    removed after that.

    NumPy Eigensolver computes up to the first :math:`k` eigenvalues of a complex-valued square
    matrix of dimension :math:`n \times n`, with :math:`k \leq n`.

    Note:
        Operators are automatically converted to SciPy's ``spmatrix``
        as needed and this conversion can be costly in terms of memory and performance as the
        operator size, mostly in terms of number of qubits it represents, gets larger.
    """

    @deprecate_func(
        additional_msg=(
            "Instead, use the class ``qiskit.algorithms.eigensolvers.NumPyEigensolver``. "
            "See https://qisk.it/algo_migration for a migration guide."
        ),
        since="0.24.0",
    )
    def __init__(
        self,
        k: int = 1,
        filter_criterion: Callable[
            [list | np.ndarray, float, ListOrDict[float] | None], bool
        ] = None,
    ) -> None:
        """
        Args:
            k: How many eigenvalues are to be computed, has a min. value of 1.
            filter_criterion: callable that allows to filter eigenvalues/eigenstates, only feasible
                eigenstates are returned in the results. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to keep this value in the final returned result or not. If the number of
                elements that satisfies the criterion is smaller than `k` then the returned list has
                fewer elements and can even be empty.
        """
        validate_min("k", k, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__()

        self._in_k = k
        self._k = k

        self._filter_criterion = filter_criterion

        self._ret = EigensolverResult()

    @property
    def k(self) -> int:
        """returns k (number of eigenvalues requested)"""
        return self._in_k

    @k.setter
    def k(self, k: int) -> None:
        """set k (number of eigenvalues requested)"""
        validate_min("k", k, 1)
        self._in_k = k
        self._k = k

    @property
    def filter_criterion(
        self,
    ) -> Callable[[list | np.ndarray, float, ListOrDict[float] | None], bool] | None:
        """returns the filter criterion if set"""
        return self._filter_criterion

    @filter_criterion.setter
    def filter_criterion(
        self,
        filter_criterion: Callable[[list | np.ndarray, float, ListOrDict[float] | None], bool]
        | None,
    ) -> None:
        """set the filter criterion"""
        self._filter_criterion = filter_criterion

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _check_set_k(self, operator: OperatorBase) -> None:
        if operator is not None:
            if self._in_k > 2**operator.num_qubits:
                self._k = 2**operator.num_qubits
                logger.debug(
                    "WARNING: Asked for %s eigenvalues but max possible is %s.", self._in_k, self._k
                )
            else:
                self._k = self._in_k

    def _solve(self, operator: OperatorBase) -> None:
        sp_mat = operator.to_spmatrix()
        # If matrix is diagonal, the elements on the diagonal are the eigenvalues. Solve by sorting.
        if scisparse.csr_matrix(sp_mat.diagonal()).nnz == sp_mat.nnz:
            diag = sp_mat.diagonal()
            indices = np.argsort(diag)[: self._k]
            eigval = diag[indices]
            eigvec = np.zeros((sp_mat.shape[0], self._k))
            for i, idx in enumerate(indices):
                eigvec[idx, i] = 1.0
        else:
            if self._k >= 2**operator.num_qubits - 1:
                logger.debug("SciPy doesn't support to get all eigenvalues, using NumPy instead.")
                if operator.is_hermitian():
                    eigval, eigvec = np.linalg.eigh(operator.to_matrix())
                else:
                    eigval, eigvec = np.linalg.eig(operator.to_matrix())
            else:
                if operator.is_hermitian():
                    eigval, eigvec = scisparse.linalg.eigsh(sp_mat, k=self._k, which="SA")
                else:
                    eigval, eigvec = scisparse.linalg.eigs(sp_mat, k=self._k, which="SR")
            indices = np.argsort(eigval)[: self._k]
            eigval = eigval[indices]
            eigvec = eigvec[:, indices]
        self._ret.eigenvalues = eigval
        self._ret.eigenstates = eigvec.T

    def _get_ground_state_energy(self, operator: OperatorBase) -> None:
        if self._ret.eigenvalues is None or self._ret.eigenstates is None:
            self._solve(operator)

    def _get_energies(
        self, operator: OperatorBase, aux_operators: ListOrDict[OperatorBase] | None
    ) -> None:
        if self._ret.eigenvalues is None or self._ret.eigenstates is None:
            self._solve(operator)

        if aux_operators is not None:
            aux_op_vals = []
            for i in range(self._k):
                aux_op_vals.append(
                    self._eval_aux_operators(aux_operators, self._ret.eigenstates[i])
                )
            self._ret.aux_operator_eigenvalues = aux_op_vals

    @staticmethod
    def _eval_aux_operators(
        aux_operators: ListOrDict[OperatorBase], wavefn, threshold: float = 1e-12
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
            value = 0.0
            if operator.coeff != 0:
                mat = operator.to_spmatrix()
                # Terra doesn't support sparse yet, so do the matmul directly if so
                # This is necessary for the particle_hole and other chemistry tests because the
                # pauli conversions are 2^12th large and will OOM error if not sparse.
                if isinstance(mat, scisparse.spmatrix):
                    value = mat.dot(wavefn).dot(np.conj(wavefn))
                else:
                    value = StateFn(operator, is_measurement=True).eval(wavefn)
                value = value if np.abs(value) > threshold else 0.0
            # The value get's wrapped into a tuple: (mean, standard deviation).
            # Since this is an exact computation, the standard deviation is known to be zero.
            values[key] = (value, 0.0)
        return values

    def compute_eigenvalues(
        self, operator: OperatorBase, aux_operators: ListOrDict[OperatorBase] | None = None
    ) -> EigensolverResult:
        super().compute_eigenvalues(operator, aux_operators)

        if operator is None:
            raise AlgorithmError("Operator was never provided")

        self._check_set_k(operator)
        zero_op = I.tensorpower(operator.num_qubits) * 0.0
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

        self._ret = EigensolverResult()
        self._solve(operator)

        # compute energies before filtering, as this also evaluates the aux operators
        self._get_energies(operator, aux_operators)

        # if a filter is set, loop over the given values and only keep
        if self._filter_criterion:

            eigvecs = []
            eigvals = []
            aux_ops = []
            cnt = 0
            for i in range(len(self._ret.eigenvalues)):
                eigvec = self._ret.eigenstates[i]
                eigval = self._ret.eigenvalues[i]
                if self._ret.aux_operator_eigenvalues is not None:
                    aux_op = self._ret.aux_operator_eigenvalues[i]
                else:
                    aux_op = None
                if self._filter_criterion(eigvec, eigval, aux_op):
                    cnt += 1
                    eigvecs += [eigvec]
                    eigvals += [eigval]
                    if self._ret.aux_operator_eigenvalues is not None:
                        aux_ops += [aux_op]
                if cnt == k_orig:
                    break

            self._ret.eigenstates = np.array(eigvecs)
            self._ret.eigenvalues = np.array(eigvals)
            # conversion to np.array breaks in case of aux_ops
            self._ret.aux_operator_eigenvalues = aux_ops

            self._k = k_orig

        # evaluate ground state after filtering (in case a filter is set)
        self._get_ground_state_energy(operator)
        if self._ret.eigenstates is not None:
            self._ret.eigenstates = ListOp([StateFn(vec) for vec in self._ret.eigenstates])

        logger.debug("EigensolverResult:\n%s", self._ret)
        return self._ret
