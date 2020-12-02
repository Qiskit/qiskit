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

""" EvolutionOp Class """

import logging
from typing import List, Optional, Set, Union, cast

import numpy as np
import scipy
from qiskit.circuit import Instruction, ParameterExpression

from ..list_ops import ListOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.tensored_op import TensoredOp
from ..operator_base import OperatorBase
from ..primitive_ops.matrix_op import MatrixOp
from ..primitive_ops.primitive_op import PrimitiveOp

logger = logging.getLogger(__name__)


class EvolvedOp(PrimitiveOp):
    r"""
    Class for wrapping Operator Evolutions for compilation (``convert``) by an EvolutionBase
    method later, essentially acting as a placeholder. Note that EvolvedOp is a weird case of
    PrimitiveOp. It happens to be that it fits into the PrimitiveOp interface nearly perfectly,
    and it essentially represents a placeholder for a PrimitiveOp later, even though it doesn't
    actually hold a primitive object. We could have chosen for it to be an OperatorBase,
    but would have ended up copying and pasting a lot of code from PrimitiveOp."""

    def __init__(self,
                 primitive: OperatorBase,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
        Args:
            primitive: The operator being wrapped to signify evolution later.
            coeff: A coefficient multiplying the operator
        """
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return self.primitive.primitive_strings()  # type: ignore

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits  # type: ignore

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, EvolvedOp) and self.primitive == other.primitive:
            return EvolvedOp(self.primitive, coeff=self.coeff + other.coeff)  # type: ignore

        if isinstance(other, SummedOp):
            op_list = [self] + cast(SummedOp, other).oplist  # type: ignore
            return SummedOp(op_list)  # type: ignore

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return EvolvedOp(
            self.primitive.adjoint() * -1,  # type: ignore
            coeff=self.coeff.conjugate()
        )

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, EvolvedOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, TensoredOp):
            return TensoredOp([self] + other.oplist)  # type: ignore

        return TensoredOp([self, other])

    def _expand_dim(self, num_qubits: int) -> OperatorBase:
        from qiskit.aqua.operators import I

        return self.tensor(I ^ num_qubits)

    def permute(self, permutation: List[int]) -> OperatorBase:
        return EvolvedOp(self.primitive.permute(permutation), coeff=self.coeff)  # type: ignore

    def compose(self, other: OperatorBase,
                permutation: Optional[List[int]] = None, front: bool = False) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        if front:
            return other.compose(new_self)
        if isinstance(other, ComposedOp):
            return ComposedOp([new_self] + other.oplist)

        return ComposedOp([new_self, other])

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return 'e^(-i*{})'.format(prim_str)
        else:
            return "{} * e^(-i*{})".format(self.coeff, prim_str)

    def __repr__(self) -> str:
        return "EvolvedOp({}, coeff={})".format(repr(self.primitive), self.coeff)

    def reduce(self) -> OperatorBase:
        return EvolvedOp(self.primitive.reduce(), coeff=self.coeff)  # type: ignore

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return EvolvedOp(
            self.primitive.bind_parameters(param_dict), coeff=param_value)  # type: ignore

    def eval(self,
             front: Optional[Union[str, dict, np.ndarray,
                                   OperatorBase]] = None) -> Union[OperatorBase, float, complex]:
        return cast(Union[OperatorBase, float, complex], self.to_matrix_op().eval(front=front))

    def to_matrix(self, massive: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        if self.primitive.__class__.__name__ == ListOp.__name__:
            return [
                op.exp_i().to_matrix(massive=massive)
                * self.primitive.coeff  # type: ignore
                * self.coeff
                for op in self.primitive.oplist  # type: ignore
            ]

        prim_mat = -1.j * self.primitive.to_matrix()  # type: ignore
        # pylint: disable=no-member
        return scipy.linalg.expm(prim_mat) * self.coeff

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Returns a ``MatrixOp`` equivalent to this Operator. """
        if self.primitive.__class__.__name__ == ListOp.__name__:
            return ListOp(
                [op.exp_i().to_matrix_op() for op in self.primitive.oplist],  # type: ignore
                coeff=self.primitive.coeff * self.coeff)  # type: ignore

        prim_mat = EvolvedOp(self.primitive).to_matrix(massive=massive)  # type: ignore
        return MatrixOp(prim_mat, coeff=self.coeff)

    def log_i(self, massive: bool = False) -> OperatorBase:
        return self.primitive * self.coeff  # type: ignore

    # pylint: disable=arguments-differ
    def to_instruction(self, massive: bool = False) -> Instruction:
        return self.primitive.to_matrix_op(massive=massive).exp_i()  # type: ignore
