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

""" CircuitOp Class """

from typing import Union, Optional, Set, List, cast
import logging
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.circuit.library import IGate
from qiskit.circuit import Instruction, ParameterExpression

from ..operator_base import OperatorBase
from ..list_ops.summed_op import SummedOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.tensored_op import TensoredOp
from .primitive_op import PrimitiveOp

logger = logging.getLogger(__name__)


class CircuitOp(PrimitiveOp):
    """ Class for Operators backed by Terra's ``QuantumCircuit`` module.

    """

    def __init__(self,
                 primitive: Union[Instruction, QuantumCircuit] = None,
                 coeff: Optional[Union[int, float, complex,
                                       ParameterExpression]] = 1.0) -> None:
        """
        Args:
            primitive: The QuantumCircuit which defines the
            behavior of the underlying function.
            coeff: A coefficient multiplying the primitive

        Raises:
            TypeError: Unsupported primitive, or primitive has ClassicalRegisters.
        """
        if isinstance(primitive, Instruction):
            qc = QuantumCircuit(primitive.num_qubits)
            qc.append(primitive, qargs=range(primitive.num_qubits))
            primitive = qc

        if not isinstance(primitive, QuantumCircuit):
            raise TypeError('CircuitOp can only be instantiated with '
                            'QuantumCircuit, not {}'.format(type(primitive)))

        if len(primitive.clbits) != 0:
            raise TypeError('CircuitOp does not support QuantumCircuits with ClassicalRegisters.')

        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {'QuantumCircuit'}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits  # type: ignore

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over operators with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        if isinstance(other, CircuitOp) and self.primitive == other.primitive:
            return CircuitOp(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return CircuitOp(self.primitive.inverse(), coeff=np.conj(self.coeff))  # type: ignore

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, CircuitOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp
        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            other = other.to_circuit_op()

        if isinstance(other, CircuitOp):
            new_qc = QuantumCircuit(self.num_qubits + other.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(other.to_instruction(),
                          qargs=new_qc.qubits[0:other.primitive.num_qubits])  # type: ignore
            new_qc.append(self.to_instruction(),
                          qargs=new_qc.qubits[other.primitive.num_qubits:])  # type: ignore
            new_qc = new_qc.decompose()
            return CircuitOp(new_qc, coeff=self.coeff * other.coeff)

        return TensoredOp([self, other])

    def compose(self, other: OperatorBase) -> OperatorBase:
        other = self._check_zero_for_composition_and_expand(other)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from ..operator_globals import Zero
        from ..state_fns import CircuitStateFn
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp

        if other == Zero ^ self.num_qubits:
            return CircuitStateFn(self.primitive, coeff=self.coeff)

        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            other = other.to_circuit_op()

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            new_qc = other.primitive.compose(self.primitive)  # type: ignore
            if isinstance(other, CircuitStateFn):
                return CircuitStateFn(new_qc,
                                      is_measurement=other.is_measurement,
                                      coeff=self.coeff * other.coeff)
            else:
                return CircuitOp(new_qc, coeff=self.coeff * other.coeff)

        return ComposedOp([self, other])

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # NOTE: not reversing qubits!! We generally reverse endianness when converting between
        # circuit or Pauli representation and matrix representation, but we don't need to here
        # because the Unitary simulator already presents the endianness of the circuit unitary in
        # forward endianness.
        unitary_backend = BasicAer.get_backend('unitary_simulator')
        unitary = execute(self.to_circuit(),
                          unitary_backend,
                          optimization_level=0).result().get_unitary()
        # pylint: disable=cyclic-import
        from ..operator_globals import EVAL_SIG_DIGITS
        return np.round(unitary * self.coeff, decimals=EVAL_SIG_DIGITS)

    def __str__(self) -> str:
        qc = self.reduce().to_circuit()  # type: ignore
        prim_str = str(qc.draw(output='text'))
        if self.coeff == 1.0:
            return prim_str
        else:
            return "{} * {}".format(self.coeff, prim_str)

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.parameters:  # type: ignore
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                # pylint: disable=import-outside-toplevel
                from ..list_ops.list_op import ListOp
                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if isinstance(self.coeff, ParameterExpression) \
                    and self.coeff.parameters <= set(unrolled_dict.keys()):
                param_instersection = set(unrolled_dict.keys()) & self.coeff.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                param_value = float(self.coeff.bind(binds))
            # & is set intersection, check if any parameters in unrolled are present in circuit
            # This is different from bind_parameters in Terra because they check for set equality
            if set(unrolled_dict.keys()) & self.primitive.parameters:  # type: ignore
                # Only bind the params found in the circuit
                param_instersection = \
                    set(unrolled_dict.keys()) & self.primitive.parameters  # type: ignore
                binds = {param: unrolled_dict[param] for param in param_instersection}
                qc = self.to_circuit().assign_parameters(binds)
        return self.__class__(qc, coeff=param_value)

    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        # pylint: disable=import-outside-toplevel
        from ..state_fns import CircuitStateFn
        from ..list_ops import ListOp
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitOp, MatrixOp, CircuitStateFn)):
            return self.compose(front)

        return cast(Union[OperatorBase, float, complex], self.to_matrix_op().eval(front=front))

    def to_circuit(self) -> QuantumCircuit:
        return self.primitive

    def to_circuit_op(self) -> OperatorBase:
        return self

    def to_instruction(self) -> Instruction:
        return self.primitive.to_instruction()  # type: ignore

    # Warning - modifying immutable object!!
    def reduce(self) -> OperatorBase:
        if self.primitive.data is not None:  # type: ignore
            # Need to do this from the end because we're deleting items!
            for i in reversed(range(len(self.primitive.data))):  # type: ignore
                [gate, _, _] = self.primitive.data[i]  # type: ignore
                # Check if Identity or empty instruction (need to check that type is exactly
                # Instruction because some gates have lazy gate.definition population)
                # pylint: disable=unidiomatic-typecheck
                if isinstance(gate, IGate) or (type(gate) == Instruction and
                                               gate.definition.data == []):
                    del self.primitive.data[i]  # type: ignore
        return self

    def permute(self, permutation: List[int]) -> 'CircuitOp':
        r"""
        Permute the qubits of the circuit.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j of the circuit should be permuted to position permutation[j].

        Returns:
            A new CircuitOp containing the permuted circuit.
        """
        new_qc = QuantumCircuit(self.num_qubits).compose(self.primitive, qubits=permutation)
        return CircuitOp(new_qc, coeff=self.coeff)
