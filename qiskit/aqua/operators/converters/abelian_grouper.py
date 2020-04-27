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

""" AbelianGrouper Class """

import logging
import itertools
import networkx as nx

from qiskit.aqua import AquaError
from ..operator_base import OperatorBase
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp
from ..state_fns.operator_state_fn import OperatorStateFn
from ..primitive_ops.pauli_op import PauliOp
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class AbelianGrouper(ConverterBase):
    """
    The AbelianGrouper converts SummedOps into a sum of Abelian sums. Meaning,
    it will traverse the Operator, and when it finds a SummedOp, it will evaluate which of the
    summed sub-Operators commute with one another. It will then convert each of the groups of
    commuting Operators into their own SummedOps, and return the sum-of-commuting-SummedOps.
    This is particularly useful for cases where mutually commuting groups can be handled
    similarly, as in the case of Pauli Expectations, where commuting Paulis have the same
    diagonalizing circuit rotation, or Pauli Evolutions, where commuting Paulis can be
    diagonalized together. """
    def __init__(self, traverse: bool = True) -> None:
        """
        Args:
            traverse: Whether to convert only the Operator passed to ``convert``, or traverse
                down that Operator.
        """
        self._traverse = traverse

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Check if operator is a SummedOp, in which case covert it into a sum of mutually
        commuting sums, or if the Operator contains sub-Operators and ``traverse`` is True,
        attempt to convert any sub-Operators.

        Args:
            operator: The Operator to attempt to convert.

        Returns:
            The converted Operator.
        """
        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..evolutions.evolved_op import EvolvedOp

        if isinstance(operator, ListOp):
            if isinstance(operator, SummedOp) and all([isinstance(op, PauliOp)
                                                       for op in operator.oplist]):
                # For now, we only support graphs over Paulis.
                return self.group_subops(operator)
            elif self._traverse:
                return operator.traverse(self.convert)
            else:
                return operator
        elif isinstance(operator, OperatorStateFn) and self._traverse:
            return OperatorStateFn(self.convert(operator.primitive),
                                   is_measurement=operator.is_measurement,
                                   coeff=operator.coeff)
        elif isinstance(operator, EvolvedOp) and self._traverse:
            return EvolvedOp(self.convert(operator.primitive), coeff=operator.coeff)
        else:
            return operator

    def group_subops(self, list_op: ListOp) -> ListOp:
        """ Given a ListOp, attempt to group into Abelian ListOps of the same type.

        Args:
            list_op: The Operator to group into Abelian groups

        Returns:
            The grouped Operator.

        Raises:
            AquaError: Any of list_op's sub-ops do not have a ``commutes`` method.
        """
        if any([not hasattr(op, 'commutes') for op in list_op.oplist]):
            raise AquaError('Cannot determine Abelian groups if an Operator in list_op does not '
                            'contain a `commutes` method'.format())

        commutation_graph = nx.Graph()
        commutation_graph.add_nodes_from(list_op.oplist)
        commutation_graph.add_edges_from(filter(lambda ops: not ops[0].commutes(ops[1]),
                                                itertools.combinations(list_op.oplist, 2)))

        # Keys in coloring_dict are nodes, values are colors
        # pylint: disable=no-member
        coloring_dict = nx.coloring.greedy_color(commutation_graph, strategy='largest_first')

        groups = {}
        for op, color in coloring_dict.items():
            groups.setdefault(color, []).append(op)

        group_ops = [list_op.__class__(group, abelian=True) for group in groups.values()]
        if len(group_ops) == 1:
            return group_ops[0] * list_op.coeff
        else:
            return list_op.__class__(group_ops, coeff=list_op.coeff)
