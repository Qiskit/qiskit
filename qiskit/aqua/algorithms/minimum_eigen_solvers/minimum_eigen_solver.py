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

"""The Minimum Eigensolver interface"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import numpy as np
from qiskit.aqua.algorithms import AlgorithmResult
from qiskit.aqua.operators import OperatorBase, LegacyBaseOperator


class MinimumEigensolver(ABC):
    """The Minimum Eigensolver Interface.

    Algorithms that can compute a minimum eigenvalue for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_minimum_eigenvalue(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                        LegacyBaseOperator]]]] = None
    ) -> 'MinimumEigensolverResult':
        """
        Computes minimum eigenvalue. Operator and aux_operators can be supplied here and
        if not None will override any already set into algorithm so it can be reused with
        different operators. While an operator is required by algorithms, aux_operators
        are optional. To 'remove' a previous aux_operators array use an empty list here.

        Args:
            operator: If not None replaces operator in algorithm
            aux_operators:  If not None replaces aux_operators in algorithm

        Returns:
            MinimumEigensolverResult
        """
        if operator is not None:
            self.operator = operator  # type: ignore
        if aux_operators is not None:
            self.aux_operators = aux_operators if aux_operators else None  # type: ignore
        return MinimumEigensolverResult()

    def supports_aux_operators(self) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

        If the minimum eigensolver computes an eigenstate of the main operator then it
        can compute the expectation value of the aux_operators for that state. Otherwise
        they will be ignored.

        Returns:
            True if aux_operator expectations can be evaluated, False otherwise
        """
        return False

    @property  # type: ignore
    @abstractmethod
    def operator(self) -> Optional[Union[OperatorBase, LegacyBaseOperator]]:
        """Return the operator."""
        raise NotImplementedError

    @operator.setter  # type: ignore
    @abstractmethod
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """Set the operator."""
        raise NotImplementedError

    @property  # type: ignore
    @abstractmethod
    def aux_operators(self) -> Optional[List[Optional[OperatorBase]]]:
        """Returns the auxiliary operators."""
        raise NotImplementedError

    @aux_operators.setter  # type: ignore
    @abstractmethod
    def aux_operators(self,
                      aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                                  LegacyBaseOperator]]]]) -> None:
        """Set the auxiliary operators."""
        raise NotImplementedError


class MinimumEigensolverResult(AlgorithmResult):
    """ Minimum Eigensolver Result."""

    @property
    def eigenvalue(self) -> Union[None, complex]:
        """ returns eigen value """
        return self.get('eigenvalue')

    @eigenvalue.setter
    def eigenvalue(self, value: complex) -> None:
        """ set eigen value """
        self.data['eigenvalue'] = value

    @property
    def eigenstate(self) -> Union[None, np.ndarray]:
        """ return eigen state """
        return self.get('eigenstate')

    @eigenstate.setter
    def eigenstate(self, value: np.ndarray) -> None:
        """ set eigen state """
        self.data['eigenstate'] = value

    @property
    def aux_operator_eigenvalues(self) -> Union[None, np.ndarray]:
        """ return aux operator eigen values """
        return self.get('aux_operator_eigenvalues')

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: np.ndarray) -> None:
        """ set aux operator eigen values """
        self.data['aux_operator_eigenvalues'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'MinimumEigensolverResult':
        """ create new object from a dictionary """
        return MinimumEigensolverResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'energy':
            warnings.warn('energy deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else value.real
        elif key == 'energies':
            warnings.warn('energies deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else [value.real]
        elif key == 'eigvals':
            warnings.warn('eigvals deprecated, use eigenvalue property.', DeprecationWarning)
            value = super().__getitem__('eigenvalue')
            return None if value is None else [value]
        elif key == 'eigvecs':
            warnings.warn('eigvecs deprecated, use eigenstate property.', DeprecationWarning)
            value = super().__getitem__('eigenstate')
            return None if value is None else [value]
        elif key == 'aux_ops':
            warnings.warn('aux_ops deprecated, use aux_operator_eigenvalues property.',
                          DeprecationWarning)
            value = super().__getitem__('aux_operator_eigenvalues')
            return None if value is None else [value]

        return super().__getitem__(key)
