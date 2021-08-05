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

""" ComposedOp Class """

from functools import partial, reduce
from typing import List, Optional, Union, cast, Dict

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.quantum_info import Statevector


class ComposedOp(ListOp):
    """A class for lazily representing compositions of Operators. Often Operators cannot be
    efficiently composed with one another, but may be manipulated further so that they can be
    composed later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be composed, and therefore if they reach a point in which they can be, such as after
    conversion to QuantumCircuits or matrices, they can be reduced by composition."""

    def __init__(
        self,
        oplist: List[OperatorBase],
        coeff: Union[complex, ParameterExpression] = 1.0,
        abelian: bool = False,
    ) -> None:
        """
        Args:
            oplist: The Operators being composed.
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.
        """
        super().__init__(oplist, combo_fn=partial(reduce, np.dot), coeff=coeff, abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return self.oplist[0].num_qubits

    @property
    def distributive(self) -> bool:
        return False

    @property
    def settings(self) -> Dict:
        """Return settings."""
        return {"oplist": self._oplist, "coeff": self._coeff, "abelian": self._abelian}

    # TODO take advantage of the mixed product property, tensorpower each element in the composition
    # def tensorpower(self, other):
    #     """ Tensor product with Self Multiple Times """
    #     raise NotImplementedError

    def to_circuit(self) -> QuantumCircuit:
        """Returns the quantum circuit, representing the composed operator.

        Returns:
            The circuit representation of the composed operator.

        Raises:
            OpflowError: for operators where a single underlying circuit can not be obtained.
        """
        # pylint: disable=cyclic-import
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..primitive_ops.primitive_op import PrimitiveOp

        circuit_op = self.to_circuit_op()
        if isinstance(circuit_op, (PrimitiveOp, CircuitStateFn)):
            return circuit_op.to_circuit()
        raise OpflowError(
            "Conversion to_circuit supported only for operators, where a single "
            "underlying circuit can be produced."
        )

    def adjoint(self) -> "ComposedOp":
        return ComposedOp([op.adjoint() for op in reversed(self.oplist)], coeff=self.coeff)

    def compose(
        self, other: OperatorBase, permutation: Optional[List[int]] = None, front: bool = False
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(ComposedOp, new_self)

        if front:
            return other.compose(new_self)
        # Try composing with last element in list
        if isinstance(other, ComposedOp):
            return ComposedOp(new_self.oplist + other.oplist, coeff=new_self.coeff * other.coeff)

        # Try composing with last element of oplist. We only try
        # this if that last element isn't itself an
        # ComposedOp, so we can tell whether composing the
        # two elements directly worked. If it doesn't,
        # continue to the final return statement below, appending other to the oplist.
        if not isinstance(new_self.oplist[-1], ComposedOp):
            comp_with_last = new_self.oplist[-1].compose(other)
            # Attempt successful
            if not isinstance(comp_with_last, ComposedOp):
                new_oplist = new_self.oplist[0:-1] + [comp_with_last]
                return ComposedOp(new_oplist, coeff=new_self.coeff)

        return ComposedOp(new_self.oplist + [other], coeff=new_self.coeff)

    def eval(
        self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]] = None
    ) -> Union[OperatorBase, complex]:
        if self._is_empty():
            return 0.0

        # pylint: disable=cyclic-import
        from ..state_fns.state_fn import StateFn

        def tree_recursive_eval(r, l_arg):
            if isinstance(r, list):
                return [tree_recursive_eval(r_op, l_arg) for r_op in r]
            else:
                return l_arg.eval(r)

        eval_list = self.oplist.copy()
        # Only one op needs to be multiplied, so just multiply the first.
        eval_list[0] = eval_list[0] * self.coeff  # type: ignore
        if front and isinstance(front, OperatorBase):
            eval_list = eval_list + [front]
        elif front:
            eval_list = [StateFn(front, is_measurement=True)] + eval_list  # type: ignore

        return reduce(tree_recursive_eval, reversed(eval_list))

    # Try collapsing list or trees of compositions into a single <Measurement | Op | State>.
    def non_distributive_reduce(self) -> OperatorBase:
        """Reduce without attempting to expand all distributive compositions.

        Returns:
            The reduced Operator.
        """
        reduced_ops = [op.reduce() for op in self.oplist]
        reduced_ops = reduce(lambda x, y: x.compose(y), reduced_ops) * self.coeff
        if isinstance(reduced_ops, ComposedOp) and len(reduced_ops.oplist) > 1:
            return reduced_ops
        else:
            return reduced_ops[0]

    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        if len(reduced_ops) == 0:
            return self.__class__([], coeff=self.coeff, abelian=self.abelian)

        def distribute_compose(l_arg, r):
            if isinstance(l_arg, ListOp) and l_arg.distributive:
                # Either ListOp or SummedOp, returns correct type
                return l_arg.__class__(
                    [distribute_compose(l_op * l_arg.coeff, r) for l_op in l_arg.oplist]
                )
            if isinstance(r, ListOp) and r.distributive:
                return r.__class__([distribute_compose(l_arg, r_op * r.coeff) for r_op in r.oplist])
            else:
                return l_arg.compose(r)

        reduced_ops = reduce(distribute_compose, reduced_ops) * self.coeff
        if isinstance(reduced_ops, ListOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)
