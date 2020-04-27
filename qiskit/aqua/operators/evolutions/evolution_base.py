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

""" EvolutionBase Class """

import logging

from ..operator_base import OperatorBase
from ..converters.converter_base import ConverterBase

logger = logging.getLogger(__name__)


class EvolutionBase(ConverterBase):
    r"""
    A base for Evolution converters.
    Evolutions are converters which traverse an Operator tree, replacing any ``EvolvedOp`` `e`
    with a Schrodinger equation-style evolution ``CircuitOp`` equalling or approximating the
    matrix exponential of -i * the Operator contained inside (`e.primitive`). The Evolutions are
    essentially implementations of Hamiltonian Simulation algorithms, including various methods
    for Trotterization.

    """

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Traverse the operator, replacing any ``EvolutionOps`` with their equivalent evolution
        ``CircuitOps``.

         Args:
             operator: The Operator to convert.

        Returns:
            The converted Operator, with ``EvolutionOps`` replaced by ``CircuitOps``.

        """
        raise NotImplementedError

    # TODO @abstractmethod
    # def error_bounds(self):
    #     """ error bounds """
    #     raise NotImplementedError
