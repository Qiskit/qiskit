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

"""The Eigensolver result."""

import warnings
from typing import Dict, Union
import numpy as np

from qiskit.aqua.algorithms import AlgorithmResult


class EigensolverResult(AlgorithmResult):
    """ Eigensolver Result."""

    @property
    def eigenvalues(self) -> Union[None, np.ndarray]:
        """ returns eigen values """
        return self.get('eigenvalues')

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """ set eigen values """
        self.data['eigenvalues'] = value

    @property
    def eigenstates(self) -> Union[None, np.ndarray]:
        """ return eigen states """
        return self.get('eigenstates')

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """ set eigen states """
        self.data['eigenstates'] = value

    @property
    def aux_operator_eigenvalues(self) -> Union[None, np.ndarray]:
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
