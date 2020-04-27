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

""" MatrixEvolution Class """

import logging

from ..operator_base import OperatorBase
from .evolution_base import EvolutionBase
from .evolved_op import EvolvedOp
from ..primitive_ops.pauli_op import PauliOp
from ..primitive_ops.matrix_op import MatrixOp
from ..list_ops.list_op import ListOp

logger = logging.getLogger(__name__)


class MatrixEvolution(EvolutionBase):
    r"""
    Performs Evolution by classical matrix exponentiation, constructing a circuit with
    ``UnitaryGates`` or ``HamiltonianGates`` containing the exponentiation of the Operator.
    """

    def convert(self, operator: OperatorBase) -> OperatorBase:
        r"""
        Traverse the operator, replacing ``EvolvedOps`` with ``CircuitOps`` containing
        ``UnitaryGates`` or ``HamiltonianGates`` (if self.coeff is a ``ParameterExpression``)
        equalling the exponentiation of -i * operator. This is done by converting the
        ``EvolvedOp.primitive`` to a ``MatrixOp`` and simply calling ``.exp_i()`` on that.

        Args:
            operator: The Operator to convert.

        Returns:
            The converted operator.
        """
        if isinstance(operator, EvolvedOp):
            if isinstance(operator.primitive, ListOp):
                return operator.primitive.to_matrix_op().exp_i() * operator.coeff
            elif isinstance(operator.primitive, (MatrixOp, PauliOp)):
                return operator.primitive.exp_i()
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert).reduce()

        return operator
