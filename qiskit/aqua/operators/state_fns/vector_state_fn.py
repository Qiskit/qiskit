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

""" VectorStateFn Class """


from typing import Union, Set
import numpy as np

from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterExpression
from qiskit.aqua import aqua_globals

from ..operator_base import OperatorBase
from .state_fn import StateFn
from ..list_ops.list_op import ListOp


class VectorStateFn(StateFn):
    """ A class for state functions and measurements which are defined in vector
    representation, and stored using Terra's ``Statevector`` class.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[list, np.ndarray, Statevector] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
        """
        Args:
            primitive: The ``Statevector``, NumPy array, or list, which defines the behavior of
                the underlying function.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator
        """
        # Lists and Numpy arrays representing statevectors are stored
        # in Statevector objects for easier handling.
        if isinstance(primitive, (np.ndarray, list)):
            primitive = Statevector(primitive)

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        return {'Vector'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.dims())

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, VectorStateFn) and self.is_measurement == other.is_measurement:
            # Covers MatrixOperator, Statevector and custom.
            return VectorStateFn((self.coeff * self.primitive) + (other.primitive * other.coeff),
                                 is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import SummedOp
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return VectorStateFn(self.primitive.conjugate(),
                             coeff=np.conj(self.coeff),
                             is_measurement=(not self.is_measurement))

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, VectorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        return self.primitive.to_operator().data * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2**self.num_qubits))

        vec = self.primitive.data * self.coeff

        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return self

    def to_circuit_op(self) -> OperatorBase:
        """ Return ``StateFnCircuit`` corresponding to this StateFn."""
        from .circuit_state_fn import CircuitStateFn
        csfn = CircuitStateFn.from_vector(self.to_matrix(massive=True)) * self.coeff  # type: ignore
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('VectorStateFn' if not self.is_measurement
                                   else 'MeasurementVector', prim_str)
        else:
            return "{}({}) * {}".format('VectorStateFn' if not self.is_measurement
                                        else 'MeasurementVector',
                                        prim_str,
                                        self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..operator_globals import EVAL_SIG_DIGITS
        from .dict_state_fn import DictStateFn
        from .operator_state_fn import OperatorStateFn
        from .circuit_state_fn import CircuitStateFn
        if isinstance(front, DictStateFn):
            return np.round(sum([v * self.primitive.data[int(b, 2)] * front.coeff  # type: ignore
                                 for (b, v) in front.primitive.items()]) * self.coeff,
                            decimals=EVAL_SIG_DIGITS)

        if isinstance(front, VectorStateFn):
            # Need to extract the element or np.array([1]) is returned.
            return np.round(np.dot(self.to_matrix(), front.to_matrix())[0],
                            decimals=EVAL_SIG_DIGITS)

        if isinstance(front, CircuitStateFn):
            # Don't reimplement logic from CircuitStateFn
            return np.conj(
                front.adjoint().eval(self.adjoint().primitive)) * self.coeff  # type: ignore

        if isinstance(front, OperatorStateFn):
            return front.adjoint().eval(self.primitive) * self.coeff  # type: ignore

        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff  # type: ignore

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        deterministic_counts = self.primitive.probabilities_dict()
        # Don't need to square because probabilities_dict already does.
        probs = np.array(list(deterministic_counts.values()))
        unique, counts = np.unique(aqua_globals.random.choice(list(deterministic_counts.keys()),
                                                              size=shots,
                                                              p=(probs / sum(probs))),
                                   return_counts=True)
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
