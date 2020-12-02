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

""" PrimitiveOp Class """

from typing import Optional, Union, Set, List, Dict
import logging
import numpy as np
from scipy.sparse import spmatrix
import scipy.linalg

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.quantum_info import Operator as MatrixOperator

from ..legacy.base_operator import LegacyBaseOperator
from ..list_ops.summed_op import SummedOp
from ..operator_base import OperatorBase

logger = logging.getLogger(__name__)


class PrimitiveOp(OperatorBase):
    r"""
    A class for representing basic Operators, backed by Operator primitives from
    Terra. This class (and inheritors) primarily serves to allow the underlying
    primitives to "flow" - i.e. interoperability and adherence to the Operator formalism
    - while the core computational logic mostly remains in the underlying primitives.
    For example, we would not produce an interface in Terra in which
    ``QuantumCircuit1 + QuantumCircuit2`` equaled the Operator sum of the circuit
    unitaries, rather than simply appending the circuits. However, within the Operator
    flow summing the unitaries is the expected behavior.

    Note that all mathematical methods are not in-place, meaning that they return a
    new object, but the underlying primitives are not copied.

    """

    @staticmethod
    # pylint: disable=unused-argument
    def __new__(cls,
                primitive:
                Optional[Union[Instruction, QuantumCircuit, List,
                               np.ndarray, spmatrix, MatrixOperator, Pauli, SparsePauliOp]] = None,
                coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> 'PrimitiveOp':
        """ A factory method to produce the correct type of PrimitiveOp subclass
        based on the primitive passed in. Primitive and coeff arguments are passed into
        subclass's init() as-is automatically by new().

        Args:
            primitive: The operator primitive being wrapped.
            coeff: A coefficient multiplying the primitive.

        Returns:
            The appropriate PrimitiveOp subclass for ``primitive``.

        Raises:
            TypeError: Unsupported primitive type passed.
        """
        if cls.__name__ != PrimitiveOp.__name__:
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (Instruction, QuantumCircuit)):
            from .circuit_op import CircuitOp
            return CircuitOp.__new__(CircuitOp)

        if isinstance(primitive, (list, np.ndarray, spmatrix, MatrixOperator)):
            from .matrix_op import MatrixOp
            return MatrixOp.__new__(MatrixOp)

        if isinstance(primitive, Pauli):
            from .pauli_op import PauliOp
            return PauliOp.__new__(PauliOp)

        if isinstance(primitive, SparsePauliOp):
            from .pauli_sum_op import PauliSumOp
            return PauliSumOp.__new__(PauliSumOp)

        raise TypeError('Unsupported primitive type {} passed into PrimitiveOp '
                        'factory constructor'.format(type(primitive)))

    def __init__(self,
                 primitive:
                 Optional[Union[Instruction, QuantumCircuit, List,
                                np.ndarray, spmatrix, MatrixOperator, Pauli, SparsePauliOp]] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
            Args:
                primitive: The operator primitive being wrapped.
                coeff: A coefficient multiplying the primitive.
        """
        self._primitive = primitive
        self._coeff = coeff

    @property
    def primitive(self) -> Union[Instruction, QuantumCircuit, List,
                                 np.ndarray, spmatrix, MatrixOperator, Pauli]:
        """ The primitive defining the underlying function of the Operator.

        Returns:
             The primitive object.
        """
        return self._primitive

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """
        The scalar coefficient multiplying the Operator.

        Returns:
              The coefficient.
        """
        return self._coeff

    @property
    def num_qubits(self) -> int:
        raise NotImplementedError

    def primitive_strings(self) -> Set[str]:
        raise NotImplementedError

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        raise NotImplementedError

    def equals(self, other: OperatorBase) -> bool:
        raise NotImplementedError

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        # Need to return self.__class__ in case the object is one of the inherited OpPrimitives
        return self.__class__(self.primitive, coeff=self.coeff * scalar)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        # Hack to make Z^(I^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other < 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        temp = PrimitiveOp(self.primitive, coeff=self.coeff)  # type: OperatorBase
        for _ in range(other - 1):
            temp = temp.tensor(self)
        return temp

    def compose(self, other: OperatorBase,
                permutation: Optional[List[int]] = None, front: bool = False) -> \
            OperatorBase:
        from ..list_ops.composed_op import ComposedOp
        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        if isinstance(other, ComposedOp):
            comp_with_first = new_self.compose(other.oplist[0])
            if not isinstance(comp_with_first, ComposedOp):
                new_oplist = [comp_with_first] + other.oplist[1:]
                return ComposedOp(new_oplist, coeff=other.coeff)
            return ComposedOp([new_self] + other.oplist, coeff=other.coeff)  # type: ignore

        return ComposedOp([new_self, other])

    def power(self, exponent: int) -> OperatorBase:
        if not isinstance(exponent, int) or exponent <= 0:
            raise TypeError('power can only take positive int arguments')
        temp = PrimitiveOp(self.primitive, coeff=self.coeff)  # type: OperatorBase
        for _ in range(exponent - 1):
            temp = temp.compose(self)
        return temp

    def _expand_dim(self, num_qubits: int) -> OperatorBase:
        raise NotImplementedError

    def permute(self, permutation: List[int]) -> OperatorBase:
        raise NotImplementedError

    def exp_i(self) -> OperatorBase:
        """ Return Operator exponentiation, equaling e^(-i * op)"""
        # pylint: disable=cyclic-import,import-outside-toplevel
        from qiskit.aqua.operators import EvolvedOp
        return EvolvedOp(self)

    def log_i(self, massive: bool = False) -> OperatorBase:
        """Return a ``MatrixOp`` equivalent to log(H)/-i for this operator H. This
        function is the effective inverse of exp_i, equivalent to finding the Hermitian
        Operator which produces self when exponentiated."""
        # pylint: disable=cyclic-import
        from ..operator_globals import EVAL_SIG_DIGITS
        from .matrix_op import MatrixOp
        return MatrixOp(np.around(scipy.linalg.logm(self.to_matrix(massive=massive)) / -1j,
                                  decimals=EVAL_SIG_DIGITS))

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{}({}, coeff={})".format(type(self).__name__, repr(self.primitive), self.coeff)

    def eval(self,
             front: Optional[Union[str, Dict[str, complex], np.ndarray, OperatorBase]] = None
             ) -> Union[OperatorBase, float, complex]:
        raise NotImplementedError

    @property
    def parameters(self):
        params = set()
        if isinstance(self.primitive, (OperatorBase, QuantumCircuit)):
            params.update(self.primitive.parameters)
        if isinstance(self.coeff, ParameterExpression):
            params.update(self.coeff.parameters)
        return params

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..list_ops.list_op import ListOp
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return self.__class__(self.primitive, coeff=param_value)

    # Nothing to collapse here.
    def reduce(self) -> OperatorBase:
        return self

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        raise NotImplementedError

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Returns a ``MatrixOp`` equivalent to this Operator. """
        # pylint: disable=import-outside-toplevel
        prim_mat = self.__class__(self.primitive).to_matrix(massive=massive)
        from .matrix_op import MatrixOp
        return MatrixOp(prim_mat, coeff=self.coeff)

    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        mat_op = self.to_matrix_op(massive=massive)
        return mat_op.to_legacy_op(massive=massive)

    def to_instruction(self) -> Instruction:
        """ Returns an ``Instruction`` equivalent to this Operator. """
        raise NotImplementedError

    def to_circuit(self) -> QuantumCircuit:
        """ Returns a ``QuantumCircuit`` equivalent to this Operator. """
        qc = QuantumCircuit(self.num_qubits)
        qc.append(self.to_instruction(), qargs=range(self.primitive.num_qubits))  # type: ignore
        return qc.decompose()

    def to_circuit_op(self) -> OperatorBase:
        """ Returns a ``CircuitOp`` equivalent to this Operator. """
        # pylint: disable=import-outside-toplevel
        from .circuit_op import CircuitOp
        if self.coeff == 0:
            return CircuitOp(QuantumCircuit(self.num_qubits), coeff=0)
        return CircuitOp(self.to_circuit(), coeff=self.coeff)

    def to_pauli_op(self, massive: bool = False) -> OperatorBase:
        """ Returns a sum of ``PauliOp`` s equivalent to this Operator. """
        mat_op = self.to_matrix_op(massive=massive)
        sparse_pauli = SparsePauliOp.from_operator(mat_op.primitive)  # type: ignore
        if not sparse_pauli.to_list():
            # pylint: disable=import-outside-toplevel
            from ..operator_globals import I
            return (I ^ self.num_qubits) * 0.0
        if len(sparse_pauli) == 1:
            label, coeff = sparse_pauli.to_list()[0]
            coeff = coeff.real if np.isreal(coeff) else coeff
            return PrimitiveOp(Pauli.from_label(label), coeff * self.coeff)

        return SummedOp(
            [
                PrimitiveOp(
                    Pauli.from_label(label),  # type: ignore
                    coeff.real if coeff == coeff.real else coeff,
                )
                for (label, coeff) in sparse_pauli.to_list()
            ],
            self.coeff,
        )
