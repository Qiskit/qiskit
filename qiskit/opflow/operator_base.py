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

"""OperatorBase Class"""

import itertools
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.mixins import StarAlgebraMixin, TensorMixin
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func


class OperatorBase(StarAlgebraMixin, TensorMixin, ABC):
    """Deprecated: A base class for all Operators: PrimitiveOps, StateFns, ListOps, etc. Operators are
    defined as functions which take one complex binary function to another. These complex binary
    functions are represented by StateFns, which are themselves a special class of Operators
    taking only the ``Zero`` StateFn to the complex binary function they represent.

    Operators can be used to construct complicated functions and computation, and serve as the
    building blocks for algorithms.

    """

    # Indentation used in string representation of list operators
    # Can be changed to use another indentation than two whitespaces
    INDENTATION = "  "

    _count = itertools.count()

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self) -> None:
        super().__init__()
        self._instance_id = next(self._count)

    @property
    @abstractmethod
    def settings(self) -> Dict:
        """Return settings of this object in a dictionary.

        You can, for example, use this ``settings`` dictionary to serialize the
        object in JSON format, if the JSON encoder you use supports all types in
        the dictionary.

        Returns:
            Object settings in a dictionary.
        """
        raise NotImplementedError

    @property
    def instance_id(self) -> int:
        """Return the unique instance id."""
        return self._instance_id

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        r"""The number of qubits over which the Operator is defined. If
        ``op.num_qubits == 5``, then ``op.eval('1' * 5)`` will be valid, but
        ``op.eval('11')`` will not.

        Returns:
            The number of qubits accepted by the Operator's underlying function.
        """
        raise NotImplementedError

    @abstractmethod
    def primitive_strings(self) -> Set[str]:
        r"""Return a set of strings describing the primitives contained in the Operator. For
        example, ``{'QuantumCircuit', 'Pauli'}``. For hierarchical Operators, such as ``ListOps``,
        this can help illuminate the primitives represented in the various recursive levels,
        and therefore which conversions can be applied.

        Returns:
            A set of strings describing the primitives contained within the Operator.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, "OperatorBase", Statevector]
        ] = None,
    ) -> Union["OperatorBase", complex]:
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

        If ``front`` is None, the matrix-representation of the operator is returned.

        Args:
            front: The bitstring, dict of bitstrings (with values being coefficients), or
                StateFn to evaluated by the Operator's underlying function, or None.

        Returns:
            The output of the Operator's evaluation function. If self is a ``StateFn``, the result
            is a float or complex. If self is an Operator (``PrimitiveOp, ComposedOp, SummedOp,
            EvolvedOp,`` etc.), the result is a StateFn.
            If ``front`` is None, the matrix-representation of the operator is returned, which
            is a ``MatrixOp`` for the operators and a ``VectorStateFn`` for state-functions.
            If either self or front contain proper
            ``ListOps`` (not ListOp subclasses), the result is an n-dimensional list of complex
            or StateFn results, resulting from the recursive evaluation by each OperatorBase
            in the ListOps.

        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self):
        r"""Try collapsing the Operator structure, usually after some type of conversion,
        e.g. trying to add Operators in a SummedOp or delete needless IGates in a CircuitOp.
        If no reduction is available, just returns self.

        Returns:
            The reduced ``OperatorBase``.
        """
        raise NotImplementedError

    @abstractmethod
    def to_matrix(self, massive: bool = False) -> np.ndarray:
        r"""Return NumPy representation of the Operator. Represents the evaluation of
        the Operator's underlying function on every combination of basis binary strings.
        Warn if more than 16 qubits to force having to set ``massive=True`` if such a
        large vector is desired.

        Returns:
              The NumPy ``ndarray`` equivalent to this Operator.
        """
        raise NotImplementedError

    @abstractmethod
    def to_matrix_op(self, massive: bool = False) -> "OperatorBase":
        """Returns a ``MatrixOp`` equivalent to this Operator."""
        raise NotImplementedError

    @abstractmethod
    def to_circuit_op(self) -> "OperatorBase":
        """Returns a ``CircuitOp`` equivalent to this Operator."""
        raise NotImplementedError

    def to_spmatrix(self) -> spmatrix:
        r"""Return SciPy sparse matrix representation of the Operator. Represents the evaluation of
        the Operator's underlying function on every combination of basis binary strings.

        Returns:
              The SciPy ``spmatrix`` equivalent to this Operator.
        """
        return csr_matrix(self.to_matrix())

    def is_hermitian(self) -> bool:
        """Return True if the operator is hermitian.

        Returns: Boolean value
        """
        return (self.to_spmatrix() != self.to_spmatrix().getH()).nnz == 0

    @staticmethod
    def _indent(lines: str, indentation: str = INDENTATION) -> str:
        """Indented representation to allow pretty representation of nested operators."""
        indented_str = indentation + lines.replace("\n", f"\n{indentation}")
        if indented_str.endswith(f"\n{indentation}"):
            indented_str = indented_str[: -len(indentation)]
        return indented_str

    # Addition / Subtraction

    @abstractmethod
    def add(self, other: "OperatorBase") -> "OperatorBase":
        r"""Return Operator addition of self and other, overloaded by ``+``.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            An ``OperatorBase`` equivalent to the sum of self and other.
        """
        raise NotImplementedError

    # Negation

    def neg(self) -> "OperatorBase":
        r"""Return the Operator's negation, effectively just multiplying by -1.0,
        overloaded by ``-``.

        Returns:
            An ``OperatorBase`` equivalent to the negation of self.
        """
        return self.mul(-1.0)

    # Adjoint

    @abstractmethod
    def adjoint(self) -> "OperatorBase":
        r"""Return a new Operator equal to the Operator's adjoint (conjugate transpose),
        overloaded by ``~``. For StateFns, this also turns the StateFn into a measurement.

        Returns:
            An ``OperatorBase`` equivalent to the adjoint of self.
        """
        raise NotImplementedError

    # Equality

    def __eq__(self, other: object) -> bool:
        r"""Overload ``==`` operation to evaluate equality between Operators.

        Args:
            other: The ``OperatorBase`` to compare to self.

        Returns:
            A bool equal to the equality of self and other.
        """
        if not isinstance(other, OperatorBase):
            return NotImplemented
        return self.equals(cast(OperatorBase, other))

    @abstractmethod
    def equals(self, other: "OperatorBase") -> bool:
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
    def mul(self, scalar: Union[complex, ParameterExpression]) -> "OperatorBase":
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

    @abstractmethod
    def tensor(self, other: "OperatorBase") -> "OperatorBase":
        r"""Return tensor product between self and other, overloaded by ``^``.
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
    def tensorpower(self, other: int) -> Union["OperatorBase", int]:
        r"""Return tensor product with self multiple times, overloaded by ``^``.

        Args:
            other: The int number of times to tensor product self with itself via ``tensorpower``.

        Returns:
            An ``OperatorBase`` equivalent to the tensorpower of self by other.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self):
        r"""Return a set of Parameter objects contained in the Operator."""
        raise NotImplementedError

    # Utility functions for parameter binding

    @abstractmethod
    def assign_parameters(
        self,
        param_dict: Dict[
            ParameterExpression,
            Union[complex, ParameterExpression, List[Union[complex, ParameterExpression]]],
        ],
    ) -> "OperatorBase":
        """Binds scalar values to any Terra ``Parameters`` in the coefficients or primitives of
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

    @abstractmethod
    def _expand_dim(self, num_qubits: int) -> "OperatorBase":
        """Expands the operator with identity operator of dimension 2**num_qubits.

        Returns:
            Operator corresponding to self.tensor(identity_operator), where dimension of identity
            operator is 2 ** num_qubits.
        """
        raise NotImplementedError

    @abstractmethod
    def permute(self, permutation: List[int]) -> "OperatorBase":
        """Permutes the qubits of the operator.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j should be permuted to position permutation[j].

        Returns:
            A new OperatorBase containing the permuted operator.

        Raises:
            OpflowError: if indices do not define a new index for each qubit.
        """
        raise NotImplementedError

    def bind_parameters(
        self,
        param_dict: Dict[
            ParameterExpression,
            Union[complex, ParameterExpression, List[Union[complex, ParameterExpression]]],
        ],
    ) -> "OperatorBase":
        r"""
        Same as assign_parameters, but maintained for consistency with QuantumCircuit in
        Terra (which has both assign_parameters and bind_parameters).
        """
        return self.assign_parameters(param_dict)

    # Mostly copied from terra, but with list unrolling added:
    @staticmethod
    def _unroll_param_dict(
        value_dict: Dict[Union[ParameterExpression, ParameterVector], Union[complex, List[complex]]]
    ) -> Union[Dict[ParameterExpression, complex], List[Dict[ParameterExpression, complex]]]:
        """Unrolls the ParameterVectors in a param_dict into separate Parameters, and unrolls
        parameterization value lists into separate param_dicts without list nesting."""
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterExpression):
                unrolled_value_dict[param] = value
            if isinstance(param, ParameterVector) and isinstance(value, (list, np.ndarray)):
                if not len(param) == len(value):
                    raise ValueError(
                        "ParameterVector {} has length {}, which differs from value list {} of "
                        "len {}".format(param, len(param), value, len(value))
                    )
                unrolled_value_dict.update(zip(param, value))
        if isinstance(list(unrolled_value_dict.values())[0], list):
            # check that all are same length
            unrolled_value_dict_list = []
            try:
                for i in range(len(list(unrolled_value_dict.values())[0])):  # type: ignore
                    unrolled_value_dict_list.append(
                        OperatorBase._get_param_dict_for_index(
                            unrolled_value_dict, i  # type: ignore
                        )
                    )
                return unrolled_value_dict_list
            except IndexError as ex:
                raise OpflowError("Parameter binding lists must all be the same length.") from ex
        return unrolled_value_dict  # type: ignore

    @staticmethod
    def _get_param_dict_for_index(unrolled_dict: Dict[ParameterExpression, List[complex]], i: int):
        """Gets a single non-list-nested param_dict for a given list index from a nested one."""
        return {k: v[i] for (k, v) in unrolled_dict.items()}

    def _expand_shorter_operator_and_permute(
        self, other: "OperatorBase", permutation: Optional[List[int]] = None
    ) -> Tuple["OperatorBase", "OperatorBase"]:
        if permutation is not None:
            other = other.permute(permutation)
        new_self = self
        if not self.num_qubits == other.num_qubits:
            # pylint: disable=cyclic-import
            from .operator_globals import Zero

            if other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                other = Zero.__class__("0" * self.num_qubits)
            elif other.num_qubits < self.num_qubits:
                other = other._expand_dim(self.num_qubits - other.num_qubits)
            elif other.num_qubits > self.num_qubits:
                new_self = self._expand_dim(other.num_qubits - self.num_qubits)
        return new_self, other

    def copy(self) -> "OperatorBase":
        """Return a deep copy of the Operator."""
        return deepcopy(self)

    # Composition

    @abstractmethod
    def compose(
        self, other: "OperatorBase", permutation: Optional[List[int]] = None, front: bool = False
    ) -> "OperatorBase":
        r"""Return Operator Composition between self and other (linear algebra-style:
        A@B(x) = A(B(x))), overloaded by ``@``.

        Note: You must be conscious of Quantum Circuit vs. Linear Algebra ordering
        conventions. Meaning, X.compose(Y)
        produces an X∘Y on qubit 0, but would produce a QuantumCircuit which looks like

            -[Y]-[X]-

        Because Terra prints circuits with the initial state at the left side of the circuit.

        Args:
            other: The ``OperatorBase`` with which to compose self.
            permutation: ``List[int]`` which defines permutation on other operator.
            front: If front==True, return ``other.compose(self)``.

        Returns:
            An ``OperatorBase`` equivalent to the function composition of self and other.
        """
        raise NotImplementedError

    @staticmethod
    def _check_massive(method: str, matrix: bool, num_qubits: int, massive: bool) -> None:
        """
        Checks if matrix or vector generated will be too large.

        Args:
            method: Name of the calling method
            matrix: True if object is matrix, otherwise vector
            num_qubits: number of qubits
            massive: True if it is ok to proceed with large matrix

        Raises:
            ValueError: Massive is False and number of qubits is greater than 16
        """
        if num_qubits > 16 and not massive and not algorithm_globals.massive:
            dim = 2**num_qubits
            if matrix:
                obj_type = "matrix"
                dimensions = f"{dim}x{dim}"
            else:
                obj_type = "vector"
                dimensions = f"{dim}"
            raise ValueError(
                f"'{method}' will return an exponentially large {obj_type}, "
                f"in this case '{dimensions}' elements. "
                "Set algorithm_globals.massive=True or the method argument massive=True "
                "if you want to proceed."
            )

    # Printing

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
