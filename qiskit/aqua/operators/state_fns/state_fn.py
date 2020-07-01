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

""" StateFn Class """

from typing import Union, Optional, Callable, Set, Dict, Tuple
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression

from ..operator_base import OperatorBase
from ..legacy.base_operator import LegacyBaseOperator


class StateFn(OperatorBase):
    r"""
    A class for representing state functions and measurements.

    State functions are defined to be complex functions over a single binary string (as
    compared to an operator, which is defined as a function over two binary strings, or a
    function taking a binary function to another binary function). This function may be
    called by the eval() method.

    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value is interpreted to represent the probability of some classical
    state (binary string) being observed from a probabilistic or quantum system represented
    by a StateFn. This leads to the equivalent definition, which is that a measurement m is
    a function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).

    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization.
    """

    @staticmethod
    # pylint: disable=unused-argument
    def __new__(cls,
                primitive: Union[str, dict, Result,
                                 list, np.ndarray, Statevector,
                                 QuantumCircuit, Instruction,
                                 OperatorBase] = None,
                coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                is_measurement: bool = False) -> 'StateFn':
        """ A factory method to produce the correct type of StateFn subclass
        based on the primitive passed in. Primitive, coeff, and is_measurement arguments
        are passed into subclass's init() as-is automatically by new().

        Args:
            primitive: The primitive which defines the behavior of the underlying State function.
            coeff: A coefficient by which the state function is multiplied.
            is_measurement: Whether the StateFn is a measurement operator

        Returns:
            The appropriate StateFn subclass for ``primitive``.

        Raises:
            TypeError: Unsupported primitive type passed.
        """

        # Prevents infinite recursion when subclasses are created
        if cls.__name__ != StateFn.__name__:
            return super().__new__(cls)

        # pylint: disable=cyclic-import,import-outside-toplevel
        if isinstance(primitive, (str, dict, Result)):
            from .dict_state_fn import DictStateFn
            return DictStateFn.__new__(DictStateFn)

        if isinstance(primitive, (list, np.ndarray, Statevector)):
            from .vector_state_fn import VectorStateFn
            return VectorStateFn.__new__(VectorStateFn)

        if isinstance(primitive, (QuantumCircuit, Instruction)):
            from .circuit_state_fn import CircuitStateFn
            return CircuitStateFn.__new__(CircuitStateFn)

        if isinstance(primitive, OperatorBase):
            from .operator_state_fn import OperatorStateFn
            return OperatorStateFn.__new__(OperatorStateFn)

        raise TypeError('Unsupported primitive type {} passed into StateFn '
                        'factory constructor'.format(type(primitive)))

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[str, dict, Result,
                                  list, np.ndarray, Statevector,
                                  QuantumCircuit, Instruction,
                                  OperatorBase] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The primitive which defines the behavior of the underlying State function.
            coeff: A coefficient by which the state function is multiplied.
            is_measurement: Whether the StateFn is a measurement operator
        """
        self._primitive = primitive
        self._is_measurement = is_measurement
        self._coeff = coeff

    @property
    def primitive(self):
        """ The primitive which defines the behavior of the underlying State function. """
        return self._primitive

    @property
    def coeff(self) -> Union[int, float, complex, ParameterExpression]:
        """ A coefficient by which the state function is multiplied. """
        return self._coeff

    @property
    def is_measurement(self) -> bool:
        """ Whether the StateFn object is a measurement Operator. """
        return self._is_measurement

    def primitive_strings(self) -> Set[str]:
        raise NotImplementedError

    @property
    def num_qubits(self) -> int:
        raise NotImplementedError

    def add(self, other: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    def adjoint(self) -> OperatorBase:
        raise NotImplementedError

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, type(self)) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar: Union[int, float, complex, ParameterExpression]) -> OperatorBase:
        if not isinstance(scalar, (int, float, complex, ParameterExpression)):
            raise ValueError('Operators can only be scalar multiplied by float or complex, not '
                             '{} of type {}.'.format(scalar, type(scalar)))

        return self.__class__(self.primitive,
                              coeff=self.coeff * scalar,
                              is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        r"""
        Return tensor product between self and other, overloaded by ``^``.
        Note: You must be conscious of Qiskit's big-endian bit printing
        convention. Meaning, Plus.tensor(Zero)
        produces a \|+⟩ on qubit 0 and a \|0⟩ on qubit 1, or \|+⟩⨂\|0⟩, but
        would produce a QuantumCircuit like

            \|0⟩--
            \|+⟩--

        Because Terra prints circuits and results with qubit 0
        at the end of the string or circuit.

        Args:
            other: The ``OperatorBase`` to tensor product with self.

        Returns:
            An ``OperatorBase`` equivalent to the tensor product of self and other.
        """
        raise NotImplementedError

    def tensorpower(self, other: int) -> Union[OperatorBase, int]:
        if not isinstance(other, int) or other <= 0:
            raise TypeError('Tensorpower can only take positive int arguments')
        temp = StateFn(self.primitive,
                       coeff=self.coeff,
                       is_measurement=self.is_measurement)  # type: OperatorBase
        for _ in range(other - 1):
            temp = temp.tensor(self)
        return temp

    def _check_zero_for_composition_and_expand(self, other: OperatorBase) \
            -> Tuple[OperatorBase, OperatorBase]:
        new_self = self
        # pylint: disable=import-outside-toplevel
        if not self.num_qubits == other.num_qubits:
            from qiskit.aqua.operators import Zero
            if self == StateFn({'0': 1}, is_measurement=True):
                # Zero is special - we'll expand it to the correct qubit number.
                new_self = StateFn('0' * self.num_qubits, is_measurement=True)
            elif other == Zero:
                # Zero is special - we'll expand it to the correct qubit number.
                other = StateFn('0' * self.num_qubits)
            else:
                raise ValueError(
                    'Composition is not defined over Operators of different dimensions, {} and {}, '
                    'respectively.'.format(self.num_qubits, other.num_qubits))

        return new_self, other

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        raise NotImplementedError

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return matrix representing product of StateFn evaluated on pairs of basis states.
        Overridden by child classes.

        Args:
            massive: Whether to allow large conversions, e.g. creating a matrix representing
                over 16 qubits.

        Returns:
            The NumPy array representing the density matrix of the State function.

        Raises:
            ValueError: If massive is set to False, and exponentially large computation is needed.
        """
        raise NotImplementedError

    def compose(self, other: OperatorBase) -> OperatorBase:
        r"""
        Composition (Linear algebra-style: A@B(x) = A(B(x))) is not well defined for states
        in the binary function model, but is well defined for measurements.

        Args:
            other: The Operator to compose with self.

        Returns:
            An Operator equivalent to the function composition of self and other.

        Raises:
            ValueError: If self is not a measurement, it cannot be composed from the right.
        """
        # TODO maybe allow outers later to produce density operators or projectors, but not yet.
        if not self.is_measurement:
            raise ValueError(
                'Composition with a Statefunction in the first operand is not defined.')

        new_self, other = self._check_zero_for_composition_and_expand(other)
        # TODO maybe include some reduction here in the subclasses - vector and Op, op and Op, etc.
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.operators import CircuitOp

        if self.primitive == {'0' * self.num_qubits: 1.0} and isinstance(other, CircuitOp):
            # Returning CircuitStateFn
            return StateFn(other.primitive, is_measurement=self.is_measurement,
                           coeff=self.coeff * other.coeff)

        from qiskit.aqua.operators import ComposedOp
        return ComposedOp([new_self, other])

    def power(self, exponent: int) -> OperatorBase:
        """ Compose with Self Multiple Times, undefined for StateFns.

        Args:
            exponent: The number of times to compose self with self.

        Raises:
            ValueError: This function is not defined for StateFns.
        """
        raise ValueError('Composition power over Statefunctions or Measurements is not defined.')

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('StateFunction' if not self.is_measurement
                                   else 'Measurement', self.coeff)
        else:
            return "{}({}) * {}".format('StateFunction' if not self.is_measurement
                                        else 'Measurement',
                                        self.coeff,
                                        prim_str)

    def __repr__(self) -> str:
        return "{}({}, coeff={}, is_measurement={})".format(self.__class__.__name__,
                                                            repr(self.primitive),
                                                            self.coeff, self.is_measurement)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        raise NotImplementedError

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
        return self.traverse(lambda x: x.assign_parameters(param_dict), coeff=param_value)

    # Try collapsing primitives where possible. Nothing to collapse here.
    def reduce(self) -> OperatorBase:
        return self

    def traverse(self,
                 convert_fn: Callable,
                 coeff: Optional[Union[int, float, complex, ParameterExpression]] = None
                 ) -> OperatorBase:
        r"""
        Apply the convert_fn to the internal primitive if the primitive is an Operator (as in
        the case of ``OperatorStateFn``). Otherwise do nothing. Used by converters.

        Args:
            convert_fn: The function to apply to the internal OperatorBase.
            coeff: A coefficient to multiply by after applying convert_fn.

        Returns:
            The converted StateFn.
        """
        if isinstance(self.primitive, OperatorBase):
            return StateFn(convert_fn(self.primitive),
                           coeff=coeff or self.coeff, is_measurement=self.is_measurement)
        else:
            return self

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a ``VectorStateFn`` for this ``StateFn``.

        Args:
            massive: Whether to allow large conversions, e.g. creating a matrix representing
                over 16 qubits.

        Returns:
            A VectorStateFn equivalent to self.
        """
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .vector_state_fn import VectorStateFn
        return VectorStateFn(self.to_matrix(massive=massive), is_measurement=self.is_measurement)

    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        raise TypeError('A StateFn cannot be represented by LegacyBaseOperator.')

    # TODO to_dict_op

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> Dict[str, Union[int, float]]:
        """ Sample the state function as a normalized probability distribution. Returns dict of
        bitstrings in order of probability, with values being probability.

        Args:
            shots: The number of samples to take to approximate the State function.
            massive: Whether to allow large conversions, e.g. creating a matrix representing
                over 16 qubits.
            reverse_endianness: Whether to reverse the endianness of the bitstrings in the return
                dict to match Terra's big-endianness.

        Returns:
            A dict containing pairs sampled strings from the State function and sampling
            frequency divided by shots.
        """
        raise NotImplementedError
