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

""" PauliOp Class """

from typing import Union, Set, Dict, cast
import logging
import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import RZGate, RYGate, RXGate, XGate, YGate, ZGate, IGate

from ..operator_base import OperatorBase
from .primitive_op import PrimitiveOp
from ..list_ops.summed_op import SummedOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.tensored_op import TensoredOp
from ..legacy.weighted_pauli_operator import WeightedPauliOperator

logger = logging.getLogger(__name__)
PAULI_GATE_MAPPING = {'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'I': IGate()}


class PauliOp(PrimitiveOp):
    """ Class for Operators backed by Terra's ``Pauli`` module.

    """

    def __init__(self,
                 primitive: Union[Pauli] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0) -> None:
        """
            Args:
                primitive: The Pauli which defines the behavior of the underlying function.
                coeff: A coefficient multiplying the primitive.

            Raises:
                TypeError: invalid parameters.
        """
        if not isinstance(primitive, Pauli):
            raise TypeError(
                'PauliOp can only be instantiated with Paulis, not {}'.format(type(primitive)))
        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {'Pauli'}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive)

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, PauliOp) and self.primitive == other.primitive:
            return PauliOp(self.primitive, coeff=self.coeff + other.coeff)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return PauliOp(self.primitive, coeff=np.conj(self.coeff))

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, PauliOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # Both Paulis
        if isinstance(other, PauliOp):
            # Copying here because Terra's Pauli kron is in-place.
            op_copy = Pauli(x=other.primitive.x, z=other.primitive.z)  # type: ignore
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            return PauliOp(op_copy.kron(self.primitive), coeff=self.coeff * other.coeff)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        if isinstance(other, CircuitOp):
            return self.to_circuit_op().tensor(other)

        return TensoredOp([self, other])

    def compose(self, other: OperatorBase) -> OperatorBase:
        other = self._check_zero_for_composition_and_expand(other)

        # If self is identity, just return other.
        if not any(self.primitive.x + self.primitive.z):  # type: ignore
            return other * self.coeff  # type: ignore

        # Both Paulis
        if isinstance(other, PauliOp):
            product, phase = Pauli.sgn_prod(self.primitive, other.primitive)
            return PrimitiveOp(product, coeff=self.coeff * other.coeff * phase)

        # pylint: disable=cyclic-import,import-outside-toplevel
        from .circuit_op import CircuitOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        if isinstance(other, (CircuitOp, CircuitStateFn)):
            return self.to_circuit_op().compose(other)

        return ComposedOp([self, other])

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix, '
                'in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff  # type: ignore

    def to_spmatrix(self) -> spmatrix:
        """ Returns SciPy sparse matrix representation of the Operator.

        Returns:
            CSR sparse matrix representation of the Operator.

        Raises:
            ValueError: invalid parameters.
        """
        return self.primitive.to_spmatrix() * self.coeff  # type: ignore

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        if front is None:
            return self.to_matrix_op()

        # pylint: disable=import-outside-toplevel,cyclic-import
        from ..state_fns.state_fn import StateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..list_ops.list_op import ListOp
        from .circuit_op import CircuitOp

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            new_front = front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                        for front_elem in front.oplist])

        else:

            if self.num_qubits != front.num_qubits:
                raise ValueError(
                    'eval does not support operands with differing numbers of qubits, '
                    '{} and {}, respectively.'.format(
                        self.num_qubits, front.num_qubits))

            if isinstance(front, DictStateFn):

                new_dict = {}  # type: Dict
                corrected_x_bits = self.primitive.x[::-1]  # type: ignore
                corrected_z_bits = self.primitive.z[::-1]  # type: ignore

                for bstr, v in front.primitive.items():
                    bitstr = np.asarray(list(bstr)).astype(np.int).astype(np.bool)
                    new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                    new_str = ''.join(map(str, 1 * new_b_str))
                    z_factor = np.product(1 - 2 * np.logical_and(bitstr, corrected_z_bits))
                    y_factor = np.product(np.sqrt(1 - 2 * np.logical_and(corrected_x_bits,
                                                                         corrected_z_bits) + 0j))
                    new_dict[new_str] = (v * z_factor * y_factor) + new_dict.get(new_str, 0)
                    new_front = StateFn(new_dict, coeff=self.coeff * front.coeff)

            elif isinstance(front, StateFn) and front.is_measurement:
                raise ValueError('Operator composed with a measurement is undefined.')

            # Composable types with PauliOp
            elif isinstance(front, (PauliOp, CircuitOp, CircuitStateFn)):
                new_front = self.compose(front)

        # Covers VectorStateFn and OperatorStateFn
            elif isinstance(front, OperatorBase):
                new_front = self.to_matrix_op().eval(front.to_matrix_op())  # type: ignore

        return new_front

    def exp_i(self) -> OperatorBase:
        """ Return a ``CircuitOp`` equivalent to e^-iH for this operator H. """
        # if only one qubit is significant, we can perform the evolution
        corrected_x = self.primitive.x[::-1]  # type: ignore
        corrected_z = self.primitive.z[::-1]  # type: ignore
        # pylint: disable=import-outside-toplevel,no-member
        sig_qubits = np.logical_or(corrected_x, corrected_z)
        if np.sum(sig_qubits) == 0:
            # e^I is just a global phase, but we can keep track of it! Should we?
            # For now, just return identity
            return PauliOp(self.primitive)
        if np.sum(sig_qubits) == 1:
            sig_qubit_index = sig_qubits.tolist().index(True)
            coeff = np.real(self.coeff) \
                if not isinstance(self.coeff, ParameterExpression) \
                else self.coeff
            # Y rotation
            if corrected_x[sig_qubit_index] and corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RYGate(coeff))
            # Z rotation
            elif corrected_z[sig_qubit_index]:
                rot_op = PrimitiveOp(RZGate(coeff))
            # X rotation
            elif corrected_x[sig_qubit_index]:
                rot_op = PrimitiveOp(RXGate(coeff))

            from ..operator_globals import I
            left_pad = I.tensorpower(sig_qubit_index)
            right_pad = I.tensorpower(self.num_qubits - sig_qubit_index - 1)
            # Need to use overloaded operators here in case left_pad == I^0
            return left_pad ^ rot_op ^ right_pad
        else:
            from ..evolutions.evolved_op import EvolvedOp
            return EvolvedOp(self)

    def commutes(self, other_op: OperatorBase) -> bool:
        """ Returns whether self commutes with other_op.

        Args:
            other_op: An ``OperatorBase`` with which to evaluate whether self commutes.

        Returns:
            A bool equaling whether self commutes with other_op

        """
        if not isinstance(other_op, PauliOp):
            return False
        # Don't use compose because parameters will break this
        self_bits = self.primitive.z + 2 * self.primitive.x  # type: ignore
        other_bits = other_op.primitive.z + 2 * other_op.primitive.x  # type: ignore
        return all((self_bits * other_bits) * (self_bits - other_bits) == 0)

    def to_circuit(self) -> QuantumCircuit:
        # If Pauli equals identity, don't skip the IGates
        is_identity = sum(self.primitive.x + self.primitive.z) == 0  # type: ignore

        # Note: Reversing endianness!!
        qc = QuantumCircuit(len(self.primitive))
        for q, pauli_str in enumerate(reversed(self.primitive.to_label())):  # type: ignore
            gate = PAULI_GATE_MAPPING[pauli_str]
            if not pauli_str == 'I' or is_identity:
                qc.append(gate, qargs=[q])
        return qc

    def to_instruction(self) -> Instruction:
        # TODO should we just do the following because performance of adding and deleting IGates
        #  doesn't matter?
        # (Reduce removes extra IGates).
        # return PrimitiveOp(self.primitive.to_instruction(), coeff=self.coeff).reduce()

        return self.to_circuit().to_instruction()

    def to_pauli_op(self, massive: bool = False) -> OperatorBase:
        return self

    def to_legacy_op(self, massive: bool = False) -> WeightedPauliOperator:
        if isinstance(self.coeff, ParameterExpression):
            try:
                coeff = float(self.coeff)
            except TypeError:
                raise TypeError('Cannot convert Operator with unbound parameter {} to Legacy '
                                'Operator'.format(self.coeff))
        else:
            coeff = cast(float, self.coeff)
        return WeightedPauliOperator(paulis=[(coeff, self.primitive)])  # type: ignore
