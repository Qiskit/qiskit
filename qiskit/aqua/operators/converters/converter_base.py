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

""" ConverterBase Class """

import logging

from abc import ABC, abstractmethod

from ..operator_base import OperatorBase

logger = logging.getLogger(__name__)


class ConverterBase(ABC):
    r"""
    Converters take an Operator and return a new Operator, generally isomorphic
    in some way with the first, but with certain desired properties. For example,
    a converter may accept ``CircuitOp`` and return a ``SummedOp`` of
    ``PauliOps`` representing the circuit unitary. Converters may not
    have polynomial space or time scaling in their operations. On the contrary, many
    converters, such as a ``MatrixExpectation`` or ``MatrixEvolution``, which convert
    ``PauliOps`` to ``MatrixOps`` internally, will require time or space exponential
    in the number of qubits unless a clever trick is known (such as the use of sparse
    matrices). """

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept the Operator and return the converted Operator

        Args:
            operator: The Operator to convert.

        Returns:
            The converted Operator.

        """
        raise NotImplementedError
