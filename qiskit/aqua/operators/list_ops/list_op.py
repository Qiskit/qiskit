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

""" ListOp Operator Class """

from functools import reduce
from typing import List, Union, Optional, Callable, Iterator, Set, Dict

import numpy as np
from scipy.sparse import spmatrix

from qiskit.circuit import ParameterExpression
from ..legacy.base_operator import LegacyBaseOperator
from ..operator_base import OperatorBase


class ListOp(OperatorBase):
    """
    A Class for manipulating List Operators, and parent class to ``SummedOp``, ``ComposedOp``,
    and ``TensoredOp``.

    List Operators are classes for storing and manipulating lists of Operators, State functions,
    or Measurements, and include some rule or ``combo_fn`` defining how the Operator functions
    of the list constituents should be combined to form to cumulative Operator function of the
    ``ListOp``. For example, a ``SummedOp`` has an addition-based ``combo_fn``, so once the
    Operators in its list are evaluated against some bitstring to produce a list of results,
    we know to add up those results to produce the final result of the ``SummedOp``'s
    evaluation. In theory, this ``combo_fn`` can be any function over classical complex values,
    but for convenience we've chosen for them to be defined over NumPy arrays and values. This way,
    large numbers of evaluations, such as after calling ``to_matrix`` on the list constituents,
    can be efficiently combined. While the combination function is defined over classical
    values, it should be understood as the operation by which each Operators' underlying
    function is combined to form the underlying Operator function of the ``ListOp``. In this
    way, the ``ListOps`` are the basis for constructing large and sophisticated Operators,
    State Functions, and Measurements in Aqua.

    The base ``ListOp`` class is particularly interesting, as its ``combo_fn`` is "the identity
    list Operation". Meaning, if we understand the ``combo_fn`` as a function from a list of
    complex values to some output, one such function is returning the list as-is. This is
    powerful for constructing compact hierarchical Operators which return many measurements in
    multiple dimensional lists.
    """

    def __init__(self,
                 oplist: List[OperatorBase],
                 combo_fn: Callable = lambda x: x,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The list of ``OperatorBases`` defining this Operator's underlying function.
            combo_fn (callable): The recombination function to combine classical results of the
                ``oplist`` Operators' eval functions (e.g. sum).
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.

            Note that the default "recombination function" lambda above is essentially the
            identity - it accepts the list of values, and returns them in a list.
        """
        self._oplist = oplist
        self._combo_fn = combo_fn
        self._coeff = coeff
        self._abelian = abelian

    @property
    def oplist(self) -> List[OperatorBase]:
        """ The list of ``OperatorBases`` defining the underlying function of this
        Operator.

        Returns:
            The Operators defining the ListOp
        """
        return self._oplist

    @property
    def combo_fn(self) -> Callable:
        """ The function defining how to combine ``oplist`` (or Numbers, or NumPy arrays) to
        produce the Operator's underlying function. For example, SummedOp's combination function
        is to add all of the Operators in ``oplist``.

        Returns:
            The combination function.
        """
        return self._combo_fn

    @property
    def abelian(self) -> bool:
        """ Whether the Operators in ``oplist`` are known to commute with one another.

        Returns:
            A bool indicating whether the ``oplist`` is Abelian.
        """
        return self._abelian

    @property
    def distributive(self) -> bool:
        """ Indicates whether the ListOp or subclass is distributive under composition.
        ListOp and SummedOp are, meaning that (opv @ op) = (opv[0] @ op + opv[1] @ op)
        (using plus for SummedOp, list for ListOp, etc.), while ComposedOp and TensoredOp
        do not behave this way.

        Returns:
            A bool indicating whether the ListOp is distributive under composition.
        """
        return True

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ The scalar coefficient multiplying the Operator.

        Returns:
            The coefficient.
        """
        return self._coeff

    def primitive_strings(self) -> Set[str]:
        return reduce(set.union, [op.primitive_strings() for op in self.oplist])

    @property
    def num_qubits(self) -> int:
        return self.oplist[0].num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if self == other:
            return self.mul(2.0)

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .summed_op import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        # TODO do this lazily? Basically rebuilds the entire tree, and ops and adjoints almost
        #  always come in pairs, so an AdjointOp holding a reference could save copying.
        if self.__class__ == ListOp:
            return ListOp([op.adjoint() for op in self.oplist],  # type: ignore
                          combo_fn=self.combo_fn, coeff=np.conj(self.coeff), abelian=self.abelian)
        return self.__class__([op.adjoint() for op in self.oplist],  # type: ignore
                              coeff=np.conj(self.coeff), abelian=self.abelian)

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = None) -> OperatorBase:
        """ Apply the convert_fn to each node in the oplist. """
        if coeff is None:
            coeff = self.coeff

        if self.__class__ == ListOp:
            return ListOp([convert_fn(op) for op in self.oplist],  # type: ignore
                          combo_fn=self.combo_fn, coeff=coeff, abelian=self.abelian)
        return self.__class__([convert_fn(op) for op in self.oplist],  # type: ignore
                              coeff=coeff, abelian=self.abelian)

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, type(self)) or not len(self.oplist) == len(other.oplist):
            return False
        # Note, ordering matters here (i.e. different list orders will return False)
        return self.coeff == other.coeff and all(
            op1 == op2 for op1, op2 in zip(self.oplist, other.oplist))

    # We need to do this because otherwise Numpy takes over scalar multiplication and wrecks it if
    # isinstance(scalar, np.number) - this started happening when we added __get_item__().
    __array_priority__ = 10000

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))
        if self.__class__ == ListOp:
            return ListOp(self.oplist, combo_fn=self.combo_fn, coeff=scalar * self.coeff,
                          abelian=self.abelian)
        return self.__class__(self.oplist, coeff=scalar * self.coeff, abelian=self.abelian)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .tensored_op import TensoredOp
        return TensoredOp([self, other])

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        # Hack to make op1^(op2^0) work as intended.
        if other == 0:
            return 1
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .tensored_op import TensoredOp
        return TensoredOp([self] * other)

    def compose(self, other: OperatorBase) -> OperatorBase:
        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self, other])

    def power(self, exponent: int) -> OperatorBase:
        if not isinstance(exponent, int) or exponent <= 0:
            raise TypeError('power can only take positive int arguments')

        # Avoid circular dependency
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .composed_op import ComposedOp
        return ComposedOp([self] * exponent)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # Combination function must be able to handle classical values
        return self.combo_fn([op.to_matrix() * self.coeff for op in self.oplist])

    def to_spmatrix(self) -> Union[spmatrix, List[spmatrix]]:
        """ Returns SciPy sparse matrix representation of the Operator.

        Returns:
            CSR sparse matrix representation of the Operator, or List thereof.
        """

        # Combination function must be able to handle classical values
        return self.combo_fn(
            [op.to_spmatrix() for op in self.oplist]) * self.coeff  # type: ignore

    def eval(self,
             front: Optional[Union[str, Dict[str, complex], 'OperatorBase']] = None
             ) -> Union['OperatorBase', float, complex, list]:
        """
        Evaluate the Operator's underlying function, either on a binary string or another Operator.
        A square binary Operator can be defined as a function taking a binary function to another
        binary function. This method returns the value of that function for a given StateFn or
        binary string. For example, ``op.eval('0110').eval('1110')`` can be seen as querying the
        Operator's matrix representation by row 6 and column 14, and will return the complex
        value at those "indices." Similarly for a StateFn, ``op.eval('1011')`` will return the
        complex value at row 11 of the vector representation of the StateFn, as all StateFns are
        defined to be evaluated from Zero implicitly (i.e. it is as if ``.eval('0000')`` is already
        called implicitly to always "indexing" from column 0).

        ListOp's eval recursively evaluates each Operator in ``oplist``,
        and combines the results using the recombination function ``combo_fn``.

        Args:
            front: The bitstring, dict of bitstrings (with values being coefficients), or
                StateFn to evaluated by the Operator's underlying function.

        Returns:
            The output of the ``oplist`` Operators' evaluation function, combined with the
            ``combo_fn``. If either self or front contain proper ``ListOps`` (not ListOp
            subclasses), the result is an n-dimensional list of complex or StateFn results,
            resulting from the recursive evaluation by each OperatorBase in the ListOps.

        Raises:
            NotImplementedError: Raised if called for a subclass which is not distributive.
            TypeError: Operators with mixed hierarchies, such as a ListOp containing both
                PrimitiveOps and ListOps, are not supported.
            NotImplementedError: Attempting to call ListOp's eval from a non-distributive subclass.

        """
        # The below code only works for distributive ListOps, e.g. ListOp and SummedOp
        if not self.distributive:
            raise NotImplementedError(r'ListOp\'s eval function is only defined for distributive '
                                      r'Listops.')

        evals = [(self.coeff * op).eval(front) for op in self.oplist]  # type: ignore
        if all(isinstance(op, OperatorBase) for op in evals):
            return self.__class__(evals)
        elif any(isinstance(op, OperatorBase) for op in evals):
            raise TypeError('Cannot handle mixed scalar and Operator eval results.')
        else:
            return self.combo_fn(evals)

    def exp_i(self) -> OperatorBase:
        """ Return an ``OperatorBase`` equivalent to an exponentiation of self * -i, e^(-i*op)."""
        # pylint: disable=unidiomatic-typecheck
        if type(self) == ListOp:
            return ListOp([op.exp_i() for op in self.oplist],  # type: ignore
                          combo_fn=self.combo_fn,
                          coeff=self.coeff)

        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import EvolvedOp
        return EvolvedOp(self)

    def log_i(self, massive: bool = False) -> OperatorBase:
        """Return a ``MatrixOp`` equivalent to log(H)/-i for this operator H. This
        function is the effective inverse of exp_i, equivalent to finding the Hermitian
        Operator which produces self when exponentiated. For proper ListOps, applies ``log_i``
        to all ops in oplist.
        """
        if self.__class__.__name__ == ListOp.__name__:
            return ListOp([op.log_i(massive=massive) for op in self.oplist],  # type: ignore
                          combo_fn=self.combo_fn,
                          coeff=self.coeff,
                          abelian=False)

        return self.to_matrix_op(massive=massive).log_i(massive=massive)  # type: ignore

    def __str__(self) -> str:
        content_string = ',\n'.join([str(op) for op in self.oplist])
        main_string = "{}([\n{}\n])".format(
            self.__class__.__name__,
            self._indent(content_string, indentation=self.INDENTATION))
        if self.abelian:
            main_string = 'Abelian' + main_string
        if self.coeff != 1.0:
            main_string = '{} * '.format(self.coeff) + main_string
        return main_string

    def __repr__(self) -> str:
        return "{}({}, coeff={}, abelian={})".format(self.__class__.__name__,
                                                     repr(self.oplist),
                                                     self.coeff,
                                                     self.abelian)

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        if isinstance(self.coeff, ParameterExpression):
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if self.coeff.parameters <= set(unrolled_dict.keys()):
                binds = {param: unrolled_dict[param] for param in self.coeff.parameters}
                param_value = float(self.coeff.bind(binds))
        return self.traverse(lambda x: x.assign_parameters(param_dict), coeff=param_value)

    def reduce(self) -> OperatorBase:
        reduced_ops = [op.reduce() for op in self.oplist]
        if self.__class__ == ListOp:
            return ListOp(reduced_ops, combo_fn=self.combo_fn, coeff=self.coeff,
                          abelian=self.abelian)
        return self.__class__(reduced_ops, coeff=self.coeff, abelian=self.abelian)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Returns an equivalent Operator composed of only NumPy-based primitives, such as
        ``MatrixOp`` and ``VectorStateFn``. """
        if self.__class__ == ListOp:
            return ListOp(
                [op.to_matrix_op(massive=massive) for op in self.oplist],  # type: ignore
                combo_fn=self.combo_fn, coeff=self.coeff, abelian=self.abelian
                ).reduce()
        return self.__class__(
            [op.to_matrix_op(massive=massive) for op in self.oplist],  # type: ignore
            coeff=self.coeff, abelian=self.abelian
            ).reduce()

    def to_circuit_op(self) -> OperatorBase:
        """ Returns an equivalent Operator composed of only QuantumCircuit-based primitives,
        such as ``CircuitOp`` and ``CircuitStateFn``. """
        # pylint: disable=cyclic-import
        from ..state_fns.operator_state_fn import OperatorStateFn
        if self.__class__ == ListOp:
            return ListOp([op.to_circuit_op()  # type: ignore
                           if not isinstance(op, OperatorStateFn) else op
                           for op in self.oplist],
                          combo_fn=self.combo_fn, coeff=self.coeff, abelian=self.abelian
                          ).reduce()
        return self.__class__([op.to_circuit_op()  # type: ignore
                               if not isinstance(op, OperatorStateFn) else op
                               for op in self.oplist],
                              coeff=self.coeff, abelian=self.abelian).reduce()

    def to_pauli_op(self, massive: bool = False) -> OperatorBase:
        """ Returns an equivalent Operator composed of only Pauli-based primitives,
        such as ``PauliOp``. """
        # pylint: disable=cyclic-import
        from ..state_fns.state_fn import StateFn
        if self.__class__ == ListOp:
            return ListOp([op.to_pauli_op(massive=massive)  # type: ignore
                           if not isinstance(op, StateFn) else op
                           for op in self.oplist],
                          combo_fn=self.combo_fn, coeff=self.coeff, abelian=self.abelian
                          ).reduce()
        return self.__class__([op.to_pauli_op(massive=massive)  # type: ignore
                               if not isinstance(op, StateFn) else op
                               for op in self.oplist],
                              coeff=self.coeff, abelian=self.abelian
                              ).reduce()

    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        mat_op = self.to_matrix_op(massive=massive).reduce()
        if isinstance(mat_op, ListOp):
            raise TypeError('A hierarchical (non-subclass) ListOp cannot be represented by '
                            'LegacyBaseOperator.')
        return mat_op.to_legacy_op(massive=massive)

    # Array operations:

    def __getitem__(self, offset: int) -> OperatorBase:
        """ Allows array-indexing style access to the Operators in ``oplist``.

        Args:
            offset: The index of ``oplist`` desired.

        Returns:
            The ``OperatorBase`` at index ``offset`` of ``oplist``.
        """
        return self.oplist[offset]

    def __iter__(self) -> Iterator:
        """ Returns an iterator over the operators in ``oplist``.

        Returns:
            An iterator over the operators in ``oplist``
        """
        return iter(self.oplist)

    def __len__(self) -> int:
        """ Length of ``oplist``.

        Returns:
            An int equal to the length of ``oplist``.
        """
        return len(self.oplist)
