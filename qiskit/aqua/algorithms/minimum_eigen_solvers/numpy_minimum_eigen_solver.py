# -*- coding: utf-8 -*-

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

from typing import List, Optional, Union, Dict, Any
import logging
import pprint

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
                                                             LegacyBaseOperator]]]] = None
                 ) -> None:
        """
        Args:
            operator: Operator instance
            aux_operators: Auxiliary operators to be evaluated at minimum eigenvalue
        """
        self._ces = NumPyEigensolver(operator, 1, aux_operators)
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

    def supports_aux_operators(self) -> bool:
        return self._ces.supports_aux_operators()

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
        result.eigenvalue = result_ces.eigenvalues[0]
        result.eigenstate = result_ces.eigenstates[0]
        if result_ces.aux_operator_eigenvalues is not None:
            if len(result_ces.aux_operator_eigenvalues) > 0:
                result.aux_operator_eigenvalues = result_ces.aux_operator_eigenvalues[0]

        logger.debug('NumPyMinimumEigensolver dict:\n%s',
                     pprint.pformat(result.data, indent=4))

        return result
