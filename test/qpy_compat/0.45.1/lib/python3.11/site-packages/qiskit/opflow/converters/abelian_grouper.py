# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""AbelianGrouper Class"""

from collections import defaultdict
from typing import List, Tuple, Union, cast

import numpy as np
import rustworkx as rx

from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.evolutions.evolved_op import EvolvedOp
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.state_fns.operator_state_fn import OperatorStateFn
from qiskit.utils.deprecation import deprecate_func


class AbelianGrouper(ConverterBase):
    """Deprecated: The AbelianGrouper converts SummedOps into a sum of Abelian sums.

    Meaning, it will traverse the Operator, and when it finds a SummedOp, it will evaluate which of
    the summed sub-Operators commute with one another. It will then convert each of the groups of
    commuting Operators into their own SummedOps, and return the sum-of-commuting-SummedOps.
    This is particularly useful for cases where mutually commuting groups can be handled
    similarly, as in the case of Pauli Expectations, where commuting Paulis have the same
    diagonalizing circuit rotation, or Pauli Evolutions, where commuting Paulis can be
    diagonalized together.
    """

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, traverse: bool = True) -> None:
        """
        Args:
            traverse: Whether to convert only the Operator passed to ``convert``, or traverse
                down that Operator.
        """
        super().__init__()
        self._traverse = traverse

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """Check if operator is a SummedOp, in which case covert it into a sum of mutually
        commuting sums, or if the Operator contains sub-Operators and ``traverse`` is True,
        attempt to convert any sub-Operators.

        Args:
            operator: The Operator to attempt to convert.

        Returns:
            The converted Operator.
        """
        if isinstance(operator, PauliSumOp):
            return self.group_subops(operator)

        if isinstance(operator, ListOp):
            if isinstance(operator, SummedOp) and all(
                isinstance(op, PauliOp) for op in operator.oplist
            ):
                # For now, we only support graphs over Paulis.
                return self.group_subops(operator)
            elif self._traverse:
                return operator.traverse(self.convert)
        elif isinstance(operator, OperatorStateFn) and self._traverse:
            return OperatorStateFn(
                self.convert(operator.primitive),
                is_measurement=operator.is_measurement,
                coeff=operator.coeff,
            )
        elif isinstance(operator, EvolvedOp) and self._traverse:
            return EvolvedOp(self.convert(operator.primitive), coeff=operator.coeff)
        return operator

    @classmethod
    def group_subops(cls, list_op: Union[ListOp, PauliSumOp]) -> ListOp:
        """Given a ListOp, attempt to group into Abelian ListOps of the same type.

        Args:
            list_op: The Operator to group into Abelian groups

        Returns:
            The grouped Operator.

        Raises:
            OpflowError: If any of list_op's sub-ops is not ``PauliOp``.
        """
        if isinstance(list_op, ListOp):
            for op in list_op.oplist:
                if not isinstance(op, PauliOp):
                    raise OpflowError(
                        "Cannot determine Abelian groups if any Operator in list_op is not "
                        f"`PauliOp`. E.g., {op} ({type(op)})"
                    )

        edges = cls._anti_commutation_graph(list_op)
        nodes = range(len(list_op))

        graph = rx.PyGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from_no_data(edges)
        # Keys in coloring_dict are nodes, values are colors
        coloring_dict = rx.graph_greedy_color(graph)
        groups = defaultdict(list)
        for idx, color in coloring_dict.items():
            groups[color].append(idx)

        if isinstance(list_op, PauliSumOp):
            primitive = list_op.primitive
            return SummedOp(
                [PauliSumOp(primitive[group], grouping_type="TPB") for group in groups.values()],
                coeff=list_op.coeff,
            )

        group_ops: List[ListOp] = [
            list_op.__class__([list_op[idx] for idx in group], abelian=True)
            for group in groups.values()
        ]
        if len(group_ops) == 1:
            return group_ops[0].mul(list_op.coeff)
        return list_op.__class__(group_ops, coeff=list_op.coeff)

    @staticmethod
    def _anti_commutation_graph(ops: Union[ListOp, PauliSumOp]) -> List[Tuple[int, int]]:
        """Create edges (i, j) if i and j are not commutable.

        Note:
            This method is applicable to only PauliOps.

        Args:
            ops: operators

        Returns:
            A list of pairs of indices of the operators that are not commutable
        """
        # convert a Pauli operator into int vector where {I: 0, X: 2, Y: 3, Z: 1}
        if isinstance(ops, PauliSumOp):
            mat1 = np.array(
                [op.primitive.paulis.z[0] + 2 * op.primitive.paulis.x[0] for op in ops],
                dtype=np.int8,
            )
        else:
            mat1 = np.array([op.primitive.z + 2 * op.primitive.x for op in ops], dtype=np.int8)

        mat2 = mat1[:, None]
        # mat3[i, j] is True if i and j are commutable with TPB
        mat3 = (((mat1 * mat2) * (mat1 - mat2)) == 0).all(axis=2)
        # return [(i, j) if mat3[i, j] is False and i < j]
        return cast(List[Tuple[int, int]], list(zip(*np.where(np.triu(np.logical_not(mat3), k=1)))))
