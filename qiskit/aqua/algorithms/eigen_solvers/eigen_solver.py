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

"""The Eigensolver interface"""

import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import numpy as np
from qiskit.aqua.algorithms import AlgorithmResult
from qiskit.aqua.operators import OperatorBase, LegacyBaseOperator


class Eigensolver(ABC):
    """The Eigensolver Interface.

    Algorithms that can compute eigenvalues for an operator
    may implement this interface to allow different algorithms to be
    used interchangeably.
    """

    @abstractmethod
    def compute_eigenvalues(
            self,
            operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
            aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                        LegacyBaseOperator]]]] = None
    ) -> 'EigensolverResult':
        """
        Computes eigenvalues. Operator and aux_operators can be supplied here and
        if not None will override any already set into algorithm so it can be reused with
        different operators. While an operator is required by algorithms, aux_operators
        are optional. To 'remove' a previous aux_operators array use an empty list here.

        Args:
            operator: If not None replaces operator in algorithm
            aux_operators:  If not None replaces aux_operators in algorithm

        Returns:
            EigensolverResult
        """
        if operator is not None:
            self.operator = operator  # type: ignore
        if aux_operators is not None:
            self.aux_operators = aux_operators if aux_operators else None  # type: ignore
        return EigensolverResult()

    @classmethod
    def supports_aux_operators(cls) -> bool:
        """Whether computing the expectation value of auxiliary operators is supported.

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


class EigensolverResult(AlgorithmResult):
    """ Eigensolver Result."""

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """ returns eigen values """
        return self.get('eigenvalues')

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """ set eigen values """
        self.data['eigenvalues'] = value

    @property
    def eigenstates(self) -> Optional[np.ndarray]:
        """ return eigen states """
        return self.get('eigenstates')

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """ set eigen states """
        self.data['eigenstates'] = value

    @property
    def aux_operator_eigenvalues(self) -> Optional[np.ndarray]:
        """ return aux operator eigen values """
        return self.get('aux_operator_eigenvalues')

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: np.ndarray) -> None:
        """ set aux operator eigen values """
        self.data['aux_operator_eigenvalues'] = value

    @staticmethod
    def from_dict(a_dict: Dict) -> 'EigensolverResult':
        """ create new object from a dictionary """
        return EigensolverResult(a_dict)

    def __getitem__(self, key: object) -> object:
        if key == 'energy':
            warnings.warn('energy deprecated, use eigenvalues property.', DeprecationWarning)
            values = super().__getitem__('eigenvalues')
            return None if values is None or values.size == 0 else values[0].real
        elif key == 'energies':
            warnings.warn('energies deprecated, use eigenvalues property.', DeprecationWarning)
            values = super().__getitem__('eigenvalues')
            return None if values is None else [x.real for x in values]
        elif key == 'eigvals':
            warnings.warn('eigvals deprecated, use eigenvalues property.', DeprecationWarning)
            return super().__getitem__('eigenvalues')
        elif key == 'eigvecs':
            warnings.warn('eigvecs deprecated, use eigenstates property.', DeprecationWarning)
            return super().__getitem__('eigenstates')
        elif key == 'aux_ops':
            warnings.warn('aux_ops deprecated, use aux_operator_eigenvalues property.',
                          DeprecationWarning)
            return super().__getitem__('aux_operator_eigenvalues')

        return super().__getitem__(key)
