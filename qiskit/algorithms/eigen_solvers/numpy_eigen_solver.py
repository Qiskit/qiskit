# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Eigensolver algorithm."""

from typing import List, Optional, Union, Dict, Any, Tuple, Callable
import logging
import pprint
import warnings
import numpy as np
from scipy import sparse as scisparse

from qiskit.utils import aqua_globals
from qiskit.opflow import OperatorBase, LegacyBaseOperator, I, StateFn, ListOp
from qiskit.utils.validation import validate_min
from .eigen_solver import Eigensolver, EigensolverResult
from ..exceptions import AlgorithmError

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name


class NumPyEigensolver(Eigensolver):
    r"""
    The NumPy Eigensolver algorithm.

    NumPy Eigensolver computes up to the first :math:`k` eigenvalues of a complex-valued square
    matrix of dimension :math:`n \times n`, with :math:`k \leq n`.

    Note:
        Operators are automatically converted to :class:`~qiskit.opflow.MatrixOperator`
        as needed and this conversion can be costly in terms of memory and performance as the
        operator size, mostly in terms of number of qubits it represents, gets larger.
    """

    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 k: int = 1,
                 aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                             LegacyBaseOperator]]]] = None,
                 filter_criterion: Callable[[Union[List, np.ndarray], float, Optional[List[float]]],
                                            bool] = None
                 ) -> None:
        """
        Args:
            operator: Operator instance. If None is supplied it must be provided later before
                run() is called. Allowing None here permits the algorithm to be configured
                and used later when operator is available, say creating an instance an letting
                application stack use this algorithm with an operator it creates.
            k: How many eigenvalues are to be computed, has a min. value of 1.
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
            filter_criterion: callable that allows to filter eigenvalues/eigenstates, only feasible
                eigenstates are returned in the results. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to keep this value in the final returned result or not. If the number of
                elements that satisfies the criterion is smaller than `k` then the returned list has
                fewer elements and can even be empty.
        """
        validate_min('k', k, 1)
        super().__init__()

        self._operator = None
        self._aux_operators = None
        self._in_k = k
        self._k = k

        self.operator = operator
        self.aux_operators = aux_operators

        self._filter_criterion = filter_criterion

        self._ret = {}  # type: Dict[str, Any]

    @property
    def random(self):
        """Return a numpy random."""
        return aqua_globals.random

    def run(self) -> Dict:
        """Execute the classical algorithm.
        Returns:
            dict: results of an algorithm.
        """

        return self._run()

    @property
    def operator(self) -> Optional[OperatorBase]:
        return self._operator

    @operator.setter
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        if isinstance(operator, LegacyBaseOperator):
            operator = operator.to_opflow()
        self._operator = operator
        self._check_set_k()

    @property
    def aux_operators(self) -> Optional[List[Optional[OperatorBase]]]:
        return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[
                          Union[OperatorBase,
                                LegacyBaseOperator,
                                List[Optional[Union[OperatorBase,
                                                    LegacyBaseOperator]]]]]) -> None:
        if aux_operators is None:
            aux_operators = []
        elif not isinstance(aux_operators, list):
            aux_operators = [aux_operators]

        if aux_operators:
            zero_op = I.tensorpower(self.operator.num_qubits) * 0.0
            converted = [op.to_opflow() if isinstance(op, LegacyBaseOperator)
                         else op for op in aux_operators]

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]

        self._aux_operators = aux_operators

    @property
    def k(self) -> int:
        """ returns k (number of eigenvalues requested) """
        return self._in_k

    @k.setter
    def k(self, k: int) -> None:
        """ set k (number of eigenvalues requested) """
        validate_min('k', k, 1)
        self._in_k = k
        self._check_set_k()

    @property
    def filter_criterion(self) -> Optional[
            Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """ returns the filter criterion if set """
        return self._filter_criterion

    @filter_criterion.setter
    def filter_criterion(self, filter_criterion: Optional[
            Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]) -> None:
        """ set the filter criterion """
        self._filter_criterion = filter_criterion

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _check_set_k(self) -> None:
        if self._operator is not None:
            if self._in_k > 2**(self._operator.num_qubits):
                self._k = 2**(self._operator.num_qubits)
                logger.debug("WARNING: Asked for %s eigenvalues but max possible is %s.",
                             self._in_k, self._k)
            else:
                self._k = self._in_k

    def _solve(self) -> None:

        sp_mat = self._operator.to_spmatrix()
        # If matrix is diagonal, the elements on the diagonal are the eigenvalues. Solve by sorting.
        if scisparse.csr_matrix(sp_mat.diagonal()).nnz == sp_mat.nnz:
            diag = sp_mat.diagonal()
            eigval = np.sort(diag)[:self._k]
            temp = np.argsort(diag)[:self._k]
            eigvec = np.zeros((sp_mat.shape[0], self._k))
            for i, idx in enumerate(temp):
                eigvec[idx, i] = 1.0
        else:
            if self._k >= 2**(self._operator.num_qubits) - 1:
                logger.debug("SciPy doesn't support to get all eigenvalues, using NumPy instead.")
                eigval, eigvec = np.linalg.eig(self._operator.to_matrix())
            else:
                eigval, eigvec = scisparse.linalg.eigs(self._operator.to_spmatrix(),
                                                       k=self._k, which='SR')
        if self._k > 1:
            idx = eigval.argsort()
            eigval = eigval[idx]
            eigvec = eigvec[:, idx]
        self._ret['eigvals'] = eigval
        self._ret['eigvecs'] = eigvec.T

    def _get_ground_state_energy(self) -> None:
        if 'eigvals' not in self._ret or 'eigvecs' not in self._ret:
            self._solve()
        if len(self._ret['eigvals']) > 0:
            self._ret['energy'] = self._ret['eigvals'][0].real
            self._ret['wavefunction'] = self._ret['eigvecs']
        else:
            self._ret['energy'] = None
            self._ret['wavefunction'] = None

    def _get_energies(self) -> None:
        if 'eigvals' not in self._ret or 'eigvecs' not in self._ret:
            self._solve()

        energies = np.empty(self._k)
        for i in range(self._k):
            energies[i] = self._ret['eigvals'][i].real
        self._ret['energies'] = energies
        if self._aux_operators:
            aux_op_vals = []
            for i in range(self._k):
                aux_op_vals.append(self._eval_aux_operators(self._ret['eigvecs'][i]))
            self._ret['aux_ops'] = aux_op_vals

    def _eval_aux_operators(self, wavefn, threshold: float = 1e-12) -> np.ndarray:
        values = []  # type: List[Tuple[float, int]]
        for operator in self._aux_operators:
            if operator is None:
                values.append(None)
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
                value = value.real if abs(value.real) > threshold else 0.0
            values.append((value, 0))
        return np.array(values, dtype=object)

    def compute_eigenvalues(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                        LegacyBaseOperator]]]] = None
    ) -> EigensolverResult:
        super().compute_eigenvalues(operator, aux_operators)
        return self._run()

    def _run(self):
        """
        Run the algorithm to compute up to the requested k number of eigenvalues.
        Returns:
            dict: Dictionary of results
        Raises:
            AlgorithmError: if no operator has been provided
        """
        if self._operator is None:
            raise AlgorithmError("Operator was never provided")

        k_orig = self._k
        if self._filter_criterion:
            # need to consider all elements if a filter is set
            self._k = 2**(self._operator.num_qubits)

        self._ret = {}
        self._solve()

        # compute energies before filtering, as this also evaluates the aux operators
        self._get_energies()

        # if a filter is set, loop over the given values and only keep
        if self._filter_criterion:

            eigvecs = []
            eigvals = []
            energies = []
            aux_ops = []
            cnt = 0
            for i in range(len(self._ret['eigvals'])):
                eigvec = self._ret['eigvecs'][i]
                eigval = self._ret['eigvals'][i]
                energy = self._ret['energies'][i]
                if 'aux_ops' in self._ret:
                    aux_op = self._ret['aux_ops'][i]
                else:
                    aux_op = None
                if self._filter_criterion(eigvec, eigval, aux_op):
                    cnt += 1
                    eigvecs += [eigvec]
                    eigvals += [eigval]
                    energies += [energy]
                    if 'aux_ops' in self._ret:
                        aux_ops += [aux_op]
                if cnt == k_orig:
                    break

            self._ret['eigvecs'] = np.array(eigvecs)
            self._ret['eigvals'] = np.array(eigvals)
            self._ret['energies'] = np.array(energies)
            # conversion to np.array breaks in case of aux_ops
            self._ret['aux_ops'] = aux_ops

            self._k = k_orig

        # evaluate ground state after filtering (in case a filter is set)
        self._get_ground_state_energy()

        logger.debug('NumPyEigensolver _run result:\n%s',
                     pprint.pformat(self._ret, indent=4))
        result = EigensolverResult()
        if 'eigvals' in self._ret:
            result.eigenvalues = self._ret['eigvals']
        if 'eigvecs' in self._ret:
            result.eigenstates = ListOp([StateFn(vec) for vec in self._ret['eigvecs']])
        if 'aux_ops' in self._ret:
            result.aux_operator_eigenvalues = self._ret['aux_ops']

        logger.debug('EigensolverResult dict:\n%s',
                     pprint.pformat(result.data, indent=4))
        return result


class ExactEigensolver(NumPyEigensolver):
    """
    The deprecated Eigensolver algorithm.
    """

    def __init__(self, operator: LegacyBaseOperator, k: int = 1,
                 aux_operators: Optional[List[LegacyBaseOperator]] = None) -> None:
        warnings.warn('Deprecated class {}, use {}.'.format('ExactEigensolver',
                                                            'NumPyEigensolver'),
                      DeprecationWarning)
        super().__init__(operator, k, aux_operators)
