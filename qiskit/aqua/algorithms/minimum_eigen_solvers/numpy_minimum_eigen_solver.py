# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Numpy Minimum Eigensolver algorithm."""

from typing import List, Optional, Union, Dict, Any, Callable
import logging
import pprint
import numpy as np

from qiskit.aqua.algorithms import ClassicalAlgorithm, NumPyEigensolver
from qiskit.aqua.operators import OperatorBase, LegacyBaseOperator
from .minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name

class NumPyMinimumEigensolver(ClassicalAlgorithm, MinimumEigensolver):
    """
    The Numpy Minimum Eigensolver algorithm.
    """

    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                             LegacyBaseOperator]]]] = None,
                 filter_criterion: Callable[[Union[List, np.ndarray], float, Optional[List[float]]],
                                            bool] = None
                 ) -> None:
        """
        Args:
            operator: Operator instance
            aux_operators: Auxiliary operators to be evaluated at minimum eigenvalue
            filter_criterion: callable that allows to filter eigenvalues/eigenstates. The minimum
                eigensolver is only searching over feasible states and returns an eigenstate that
                has the smallest eigenvalue among feasible states. The callable has the signature
                `filter(eigenstate, eigenvalue, aux_values)` and must return a boolean to indicate
                whether to consider this value or not. If there is no
                feasible element, the result can even be empty.
        """
        self._ces = NumPyEigensolver(operator=operator, k=1, aux_operators=aux_operators,
                                     filter_criterion=filter_criterion)
        # TODO remove
        self._ret = {}  # type: Dict[str, Any]

    @property
    def operator(self) -> Optional[OperatorBase]:
        return self._ces.operator

    @operator.setter
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        self._ces.operator = operator

    @property
    def aux_operators(self) -> Optional[List[Optional[OperatorBase]]]:
        return self._ces.aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                                  LegacyBaseOperator]]]]) -> None:
        self._ces.aux_operators = aux_operators

    @property
    def filter_criterion(self) -> Optional[
            Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]:
        """ returns the filter criterion if set """
        return self._ces.filter_criterion

    @filter_criterion.setter
    def filter_criterion(self, filter_criterion: Optional[
            Callable[[Union[List, np.ndarray], float, Optional[List[float]]], bool]]) -> None:
        """ set the filter criterion """
        self._ces.filter_criterion = filter_criterion

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return NumPyEigensolver.supports_aux_operators()

    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                        LegacyBaseOperator]]]] = None
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)
        return self._run()

    def _run(self) -> MinimumEigensolverResult:
        """
        Run the algorithm to compute up to the minimum eigenvalue.
        Returns:
            dict: Dictionary of results
        """
        result_ces = self._ces.run()
        self._ret = self._ces._ret  # TODO remove

        result = MinimumEigensolverResult()
        if len(result_ces.eigenvalues) > 0:
            result.eigenvalue = result_ces.eigenvalues[0]
            result.eigenstate = result_ces.eigenstates[0]
            if result_ces.aux_operator_eigenvalues is not None:
                if len(result_ces.aux_operator_eigenvalues) > 0:
                    result.aux_operator_eigenvalues = result_ces.aux_operator_eigenvalues[0]
        else:
            result.eigenvalue = None
            result.eigenstate = None
            result.aux_operator_eigenvalues = None

        logger.debug('NumPyMinimumEigensolver dict:\n%s',
                     pprint.pformat(result.data, indent=4))

        return result
