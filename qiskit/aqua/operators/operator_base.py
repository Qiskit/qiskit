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

""" OperatorBase Class """

from typing import Set, Union, Dict, Optional, List, cast
from numbers import Number
from abc import ABC, abstractmethod
import numpy as np

from qiskit.aqua import AquaError
from qiskit.circuit import ParameterExpression, ParameterVector
from .legacy.base_operator import LegacyBaseOperator


class OperatorBase(ABC):
    """ A base class for all Operators: PrimitiveOps, StateFns, ListOps, etc. Operators are
    defined as functions which take one complex binary function to another. These complex binary
    functions are represented by StateFns, which are themselves a special class of Operators
    taking only the ``Zero`` StateFn to the complex binary function they represent.

    Operators can be used to construct complicated functions and computation, and serve as the
    building blocks for algorithms in Aqua.

    """
    # Indentation used in string representation of list operators
    # Can be changed to use another indentation than two whitespaces
    INDENTATION = '  '

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        r""" The number of qubits over which the Operator is defined. If
        ``op.num_qubits == 5``, then ``op.eval('1' * 5)`` will be valid, but
        ``op.eval('11')`` will not.

        Returns:
            The number of qubits accepted by the Operator's underlying function.
        """
        raise NotImplementedError

    @abstractmethod
    def primitive_strings(self) -> Set[str]:
        r""" Return a set of strings describing the primitives contained in the Operator. For
        example, ``{'QuantumCircuit', 'Pauli'}``. For hierarchical Operators, such as ``ListOps``,
        this can help illuminate the primitives represented in the various recursive levels,
        and therefore which conversions can be applied.

        Returns:
            A set of strings describing the primitives contained within the Operator.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self,
             front: Optional[Union[str, Dict[str, complex], 'OperatorBase']] = None
             ) -> Union['OperatorBase', float, complex, list]:
        r"""
        Evaluate the Operator's underlying function, either on a binary string or another Operator.
        A square binary Operator can be defined as a function taking a binary function to another
        binary function. This method returns the value of that function for a given StateFn or
        binary string. For example, ``op.eval('0110').eval('1110')`` can be seen as querying the
        Operator's matrix representation by row 6 and column 14, and will return the complex
        value at those "indices." Similarly for a StateFn, ``op.eval('1011')`` will return the
        complex value at row 11 of the vector representation of the StateFn, as all StateFns are
        defined to be evaluated from Zero implicitly (i.e. it is as if ``.eval('0000')`` is already
        called implicitly to always "indexing" from column 0).

        Args:
            front: The bitstring, dict of bitstrings (with values being coefficients), or
                StateFn to evaluated by the Operator's underlying function.

        Returns:
            The output of the Operator's evaluation function. If self is a ``StateFn``, the result
            is a float or complex. If self is an Operator (``PrimitiveOp, ComposedOp, SummedOp,
            EvolvedOp,`` etc.), the result is a StateFn. If either self or front contain proper
            ``ListOps`` (not ListOp subclasses), the result is an n-dimensional list of complex
            or StateFn results, resulting from the recursive evaluation by each OperatorBase
            in the ListOps.

        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self):
        r""" Try collapsing the Operator structure, usually after some type of conversion,
        e.g. trying to add Operators in a SummedOp or delete needless IGates in a CircuitOp.
        If no reduction is available, just returns self.

        Returns:
            The reduced ``OperatorBase``.
        """
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self, massive: bool = False) -> np.ndarray:
        r""" Return NumPy representation of the Operator. Represents the evaluation of
        the Operator's underlying function on every combination of basis binary strings.
        Warn if more than 16 qubits to force having to set ``massive=True`` if such a
        large vector is desired.

        Returns:
              The NumPy ``ndarray`` equivalent to this Operator.
        """
        raise NotImplementedError

    @abstractmethod
    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        r""" Attempt to return the Legacy Operator representation of the Operator. If self is a
        ``SummedOp`` of ``PauliOps``, will attempt to convert to ``WeightedPauliOperator``,
        and otherwise will simply convert to ``MatrixOp`` and then to ``MatrixOperator``. The
        Legacy Operators cannot represent ``StateFns`` or proper ``ListOps`` (meaning not one of
        the ``ListOp`` subclasses), so an error will be thrown if this method is called on such
        an Operator. Also, Legacy Operators cannot represent unbound Parameter coeffs, so an error
        will be thrown if any are present in self.

        Warn if more than 16 qubits to force having to set ``massive=True`` if such a
        large vector is desired.

        Returns:
            The ``LegacyBaseOperator`` representing this Operator.

        Raises:
            TypeError: self is an Operator which cannot be represented by a ``LegacyBaseOperator``,
                such as ``StateFn``, proper (non-subclass) ``ListOp``, or an Operator with an
                unbound coeff Parameter.
        """
        raise NotImplementedError

    @staticmethod
    def _indent(lines: str, indentation: str = INDENTATION) -> str:
        """ Indented representation to allow pretty representation of nested operators. """
        indented_str = indentation + lines.replace("\n", "\n{}".format(indentation))
        if indented_str.endswith("\n{}".format(indentation)):
            indented_str = indented_str[:-len(indentation)]
        return indented_str

    # Addition / Subtraction

    def __add__(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Overload ``+`` operation for Operator addition.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to the sum of self and other.
        """
        # Hack to be able to use sum(list_of_ops) nicely,
        # because sum adds 0 to the first element of the list.
        if other == 0:
            return self

        return self.add(other)

    def __radd__(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Overload right ``+`` operation for Operator addition.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to the sum of self and other.
        """
        # Hack to be able to use sum(list_of_ops) nicely because
        # sum adds 0 to the first element of the list.
        if other == 0:
            return self

        return self.add(other)

    @abstractmethod
    def add(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Return Operator addition of self and other, overloaded by ``+``.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to the sum of self and other.
        """
        raise NotImplementedError

    def __sub__(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Overload ``-`` operation for Operator subtraction.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to self - other.
        """
        return self.add(-other)

    def __rsub__(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Overload right ``-`` operation for Operator subtraction.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to self - other.
        """
        return self.neg().add(other)

    # Negation

    def __neg__(self) -> 'OperatorBase':
        r""" Overload unary ``-`` to return Operator negation.

        Returns:
            An ``OperatorBase`` equivalent to the negation of self.
        """
        return self.neg()

    def neg(self) -> 'OperatorBase':
        r""" Return the Operator's negation, effectively just multiplying by -1.0,
        overloaded by ``-``.

        Returns:
            An ``OperatorBase`` equivalent to the negation of self.
        """
        return self.mul(-1.0)

    # Adjoint

    def __invert__(self) -> 'OperatorBase':
        r""" Overload unary ``~`` to return Operator adjoint.

        Returns:
            An ``OperatorBase`` equivalent to the adjoint of self.
        """
        return self.adjoint()

    @abstractmethod
    def adjoint(self) -> 'OperatorBase':
        r""" Return a new Operator equal to the Operator's adjoint (conjugate transpose),
        overloaded by ``~``. For StateFns, this also turns the StateFn into a measurement.

        Returns:
            An ``OperatorBase`` equivalent to the adjoint of self.
        """
        raise NotImplementedError

    # Equality

    def __eq__(self, other: object) -> bool:
        r""" Overload ``==`` operation to evaluate equality between Operators.

        Args:
            other: The ``OperatorBase`` to compare to self.

        Returns:
            A bool equal to the equality of self and other.
        """
        if not isinstance(other, OperatorBase):
            return NotImplemented
        return self.equals(cast(OperatorBase, other))

    @abstractmethod
    def equals(self, other: 'OperatorBase') -> bool:
        r"""
        Evaluate Equality between Operators, overloaded by ``==``. Only returns True if self and
        other are of the same representation (e.g. a DictStateFn and CircuitStateFn will never be
        equal, even if their vector representations are equal), their underlying primitives are
        equal (this means for ListOps, OperatorStateFns, or EvolvedOps the equality is evaluated
        recursively downwards), and their coefficients are equal.

        Args:
            other: The ``OperatorBase`` to compare to self.

        Returns:
            A bool equal to the equality of self and other.

        """
        raise NotImplementedError

    # Scalar Multiplication

    @abstractmethod
    def mul(self, scalar: Union[Number, ParameterExpression]) -> 'OperatorBase':
        r"""
        Returns the scalar multiplication of the Operator, overloaded by ``*``, including
        support for Terra's ``Parameters``, which can be bound to values later (via
        ``bind_parameters``).

        Args:
            scalar: The real or complex scalar by which to multiply the Operator,
                or the ``ParameterExpression`` to serve as a placeholder for a scalar factor.

        Returns:
            An ``OperatorBase`` equivalent to product of self and scalar.
        """
        raise NotImplementedError

    def __mul__(self, other: Number) -> 'OperatorBase':
        r""" Overload ``*`` for Operator scalar multiplication.

        Args:
            other: The real or complex scalar by which to multiply the Operator,
                or the ``ParameterExpression`` to serve as a placeholder for a scalar factor.

        Returns:
            An ``OperatorBase`` equivalent to product of self and scalar.
        """
        return self.mul(other)

    def __rmul__(self, other: Number) -> 'OperatorBase':
        r""" Overload right ``*`` for Operator scalar multiplication.

        Args:
            other: The real or complex scalar by which to multiply the Operator,
                or the ``ParameterExpression`` to serve as a placeholder for a scalar factor.

        Returns:
            An ``OperatorBase`` equivalent to product of self and scalar.
        """
        return self.mul(other)

    def __truediv__(self, other: Union[int, float, complex]) -> 'OperatorBase':
        r""" Overload ``/`` for scalar Operator division.

        Args:
            other: The real or complex scalar by which to divide the Operator,
                or the ``ParameterExpression`` to serve as a placeholder for a scalar divisor.

        Returns:
            An ``OperatorBase`` equivalent to self divided by scalar.
        """
        return self.mul(1 / other)

    def __xor__(self, other: Union['OperatorBase', int]) -> 'OperatorBase':
        r""" Overload ``^`` for tensor product or tensorpower if other is an int.

        Args:
            other: The ``OperatorBase`` to tensor product with self, or the int number of times
                to tensor self with itself via ``tensorpower``.

        Returns:
            An ``OperatorBase`` equivalent to tensor product of self and other,
                or the tensorpower of self by other.
        """
        if isinstance(other, int):
            return cast(OperatorBase, self.tensorpower(other))
        else:
            return self.tensor(other)

    def __rxor__(self, other: Union['OperatorBase', int]) -> 'OperatorBase':
        r""" Overload right ``^`` for tensor product, a hack to make (I^0)^Z work as intended.

        Args:
            other: The ``OperatorBase`` for self to tensor product with, or 1, which indicates to
                return self.

        Returns:
            An ``OperatorBase`` equivalent to the tensor product of other and self, or self.
        """
        if other == 1:
            return self
        else:
            return cast(OperatorBase, other).tensor(self)

    @abstractmethod
    def tensor(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Return tensor product between self and other, overloaded by ``^``.
        Note: You must be conscious of Qiskit's big-endian bit printing convention.
        Meaning, X.tensor(Y) produces an X on qubit 0 and an Y on qubit 1, or X⨂Y,
        but would produce a QuantumCircuit which looks like

            -[Y]-
            -[X]-

        Because Terra prints circuits and results with qubit 0 at the end of the string
        or circuit.

        Args:
            other: The ``OperatorBase`` to tensor product with self.

        Returns:
            An ``OperatorBase`` equivalent to the tensor product of self and other.
        """
        raise NotImplementedError

    @abstractmethod
    def tensorpower(self, other: int) -> Union['OperatorBase', int]:
        r""" Return tensor product with self multiple times, overloaded by ``^``.

        Args:
            other: The int number of times to tensor product self with itself via ``tensorpower``.

        Returns:
            An ``OperatorBase`` equivalent to the tensorpower of self by other.
        """
        raise NotImplementedError

    # Utility functions for parameter binding

    @abstractmethod
    def assign_parameters(self,
                          param_dict: Dict[ParameterExpression,
                                           Union[Number,
                                                 ParameterExpression,
                                                 List[Union[Number, ParameterExpression]]]]
                          ) -> 'OperatorBase':
        """ Binds scalar values to any Terra ``Parameters`` in the coefficients or primitives of
        the Operator, or substitutes one ``Parameter`` for another. This method differs from
        Terra's ``assign_parameters`` in that it also supports lists of values to assign for a
        give ``Parameter``, in which case self will be copied for each parameterization in the
        binding list(s), and all the copies will be returned in an ``OpList``. If lists of
        parameterizations are used, every ``Parameter`` in the param_dict must have the same
        length list of parameterizations.

        Args:
            param_dict: The dictionary of ``Parameters`` to replace, and values or lists of
                values by which to replace them.

        Returns:
            The ``OperatorBase`` with the ``Parameters`` in self replaced by the
            values or ``Parameters`` in param_dict. If param_dict contains parameterization lists,
            this ``OperatorBase`` is an ``OpList``.
        """
        raise NotImplementedError

    def bind_parameters(self,
                        param_dict: Dict[ParameterExpression,
                                         Union[Number,
                                               ParameterExpression,
                                               List[Union[Number, ParameterExpression]]]]
                        ) -> 'OperatorBase':
        r"""
        Same as assign_parameters, but maintained for consistency with QuantumCircuit in
        Terra (which has both assign_parameters and bind_parameters).
        """
        return self.assign_parameters(param_dict)

    # Mostly copied from terra, but with list unrolling added:
    @staticmethod
    def _unroll_param_dict(value_dict: Dict[Union[ParameterExpression, ParameterVector],
                                            Union[Number, List[Number]]]
                           ) -> Union[Dict[ParameterExpression, Number],
                                      List[Dict[ParameterExpression, Number]]]:
        """ Unrolls the ParameterVectors in a param_dict into separate Parameters, and unrolls
        parameterization value lists into separate param_dicts without list nesting. """
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterExpression):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):  # type: ignore
                    raise ValueError(
                        'ParameterVector {} has length {}, which differs from value list {} of '
                        'len {}'.format(param, len(param), value, len(value)))  # type: ignore
                unrolled_value_dict.update(zip(param, value))  # type: ignore
        if isinstance(list(unrolled_value_dict.values())[0], list):
            # check that all are same length
            unrolled_value_dict_list = []
            try:
                for i in range(len(list(unrolled_value_dict.values())[0])):  # type: ignore
                    unrolled_value_dict_list.append(
                        OperatorBase._get_param_dict_for_index(unrolled_value_dict,  # type: ignore
                                                               i))
                return unrolled_value_dict_list
            except IndexError:
                raise AquaError('Parameter binding lists must all be the same length.')
        return unrolled_value_dict  # type: ignore

    @staticmethod
    def _get_param_dict_for_index(unrolled_dict: Dict[ParameterExpression, List[Number]],
                                  i: int):
        """ Gets a single non-list-nested param_dict for a given list index from a nested one. """
        return {k: v[i] for (k, v) in unrolled_dict.items()}

    # Composition

    def __matmul__(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Overload ``@`` for Operator composition.

        Args:
            other: The ``OperatorBase`` with which to compose self.

        Returns:
            An ``OperatorBase`` equivalent to the function composition of self and other.
        """
        return self.compose(other)

    @abstractmethod
    def compose(self, other: 'OperatorBase') -> 'OperatorBase':
        r""" Return Operator Composition between self and other (linear algebra-style:
        A@B(x) = A(B(x))), overloaded by ``@``.

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like

            -[Y]-[X]-

        Because Terra prints circuits with the initial state at the left side of the circuit.

        Args:
            other: The ``OperatorBase`` with which to compose self.

        Returns:
            An ``OperatorBase`` equivalent to the function composition of self and other.
        """
        raise NotImplementedError

    @abstractmethod
    def power(self, exponent: int) -> 'OperatorBase':
        r""" Return Operator composed with self multiple times, overloaded by ``**``.

        Args:
            exponent: The int number of times to compose self with itself.

        Returns:
            An ``OperatorBase`` equivalent to self composed with itself exponent times.
        """
        raise NotImplementedError

    def __pow__(self, exponent: int) -> 'OperatorBase':
        r""" Overload ``**`` for composition power.

        Args:
            exponent: The int number of times to compose self with itself.

        Returns:
            An ``OperatorBase`` equivalent to self composed with itself exponent times.
        """
        return self.power(exponent)

    # Printing

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
