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

"""CircuitOp Class"""

from typing import Dict, List, Optional, Set, Union, cast
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.circuit.library import IGate
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Statevector
from qiskit.utils.deprecation import deprecate_func


class CircuitOp(PrimitiveOp):
    """Deprecated: Class for Operators backed by Terra's ``QuantumCircuit`` module."""

    primitive: QuantumCircuit

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: Union[Instruction, QuantumCircuit],
        coeff: Union[complex, ParameterExpression] = 1.0,
    ) -> None:
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
            raise TypeError(
                "CircuitOp can only be instantiated with "
                "QuantumCircuit, not {}".format(type(primitive))
            )

        if len(primitive.clbits) != 0:
            raise TypeError("CircuitOp does not support QuantumCircuits with ClassicalRegisters.")

        super().__init__(primitive, coeff)
        self._coeff = coeff

    def primitive_strings(self) -> Set[str]:
        return {"QuantumCircuit"}

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over operators with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        if isinstance(other, CircuitOp) and self.primitive == other.primitive:
            return CircuitOp(self.primitive, coeff=self.coeff + other.coeff)

        # Covers all else.
        # pylint: disable=cyclic-import
        from ..list_ops.summed_op import SummedOp

        return SummedOp([self, other])

    def adjoint(self) -> "CircuitOp":
        return CircuitOp(self.primitive.inverse(), coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, CircuitOp) or not self.coeff == other.coeff:
            return False

        return self.primitive == other.primitive

    def tensor(self, other: OperatorBase) -> Union["CircuitOp", TensoredOp]:
        # pylint: disable=cyclic-import
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp

        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            other = other.to_circuit_op()

        if isinstance(other, CircuitOp):
            new_qc = QuantumCircuit(self.num_qubits + other.num_qubits)
            # NOTE!!! REVERSING QISKIT ENDIANNESS HERE
            new_qc.append(
                other.to_instruction(), qargs=new_qc.qubits[0 : other.primitive.num_qubits]
            )
            new_qc.append(self.to_instruction(), qargs=new_qc.qubits[other.primitive.num_qubits :])
            new_qc = new_qc.decompose()
            return CircuitOp(new_qc, coeff=self.coeff * other.coeff)

        return TensoredOp([self, other])

    def compose(
        self, other: OperatorBase, permutation: Optional[List[int]] = None, front: bool = False
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(CircuitOp, new_self)

        if front:
            return other.compose(new_self)
        # pylint: disable=cyclic-import
        from ..operator_globals import Zero
        from ..state_fns import CircuitStateFn
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp

        if other == Zero ^ new_self.num_qubits:
            return CircuitStateFn(new_self.primitive, coeff=new_self.coeff)

        if isinstance(other, (PauliOp, CircuitOp, MatrixOp)):
            other = other.to_circuit_op()

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            new_qc = other.primitive.compose(new_self.primitive)
            if isinstance(other, CircuitStateFn):
                return CircuitStateFn(
                    new_qc, is_measurement=other.is_measurement, coeff=new_self.coeff * other.coeff
                )
            else:
                return CircuitOp(new_qc, coeff=new_self.coeff * other.coeff)

        return super(CircuitOp, new_self).compose(other)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", True, self.num_qubits, massive)
        unitary = qiskit.quantum_info.Operator(self.to_circuit()).data
        return unitary * self.coeff

    def __str__(self) -> str:
        qc = self.to_circuit()
        prim_str = str(qc.draw(output="text"))
        if self.coeff == 1.0:
            return prim_str
        else:
            return f"{self.coeff} * {prim_str}"

    def assign_parameters(self, param_dict: dict) -> OperatorBase:
        param_value = self.coeff
        qc = self.primitive
        if isinstance(self.coeff, ParameterExpression) or self.primitive.parameters:
            unrolled_dict = self._unroll_param_dict(param_dict)
            if isinstance(unrolled_dict, list):
                from ..list_ops.list_op import ListOp

                return ListOp([self.assign_parameters(param_dict) for param_dict in unrolled_dict])
            if isinstance(self.coeff, ParameterExpression) and self.coeff.parameters <= set(
                unrolled_dict.keys()
            ):
                param_instersection = set(unrolled_dict.keys()) & self.coeff.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                param_value = float(self.coeff.bind(binds))
            # & is set intersection, check if any parameters in unrolled are present in circuit
            # This is different from bind_parameters in Terra because they check for set equality
            if set(unrolled_dict.keys()) & self.primitive.parameters:
                # Only bind the params found in the circuit
                param_instersection = set(unrolled_dict.keys()) & self.primitive.parameters
                binds = {param: unrolled_dict[param] for param in param_instersection}
                qc = self.to_circuit().assign_parameters(binds)
        return self.__class__(qc, coeff=param_value)

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        from ..state_fns import CircuitStateFn
        from ..list_ops import ListOp
        from .pauli_op import PauliOp
        from .matrix_op import MatrixOp

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        # Composable with circuit
        if isinstance(front, (PauliOp, CircuitOp, MatrixOp, CircuitStateFn)):
            return self.compose(front)

        return self.to_matrix_op().eval(front)

    def to_circuit(self) -> QuantumCircuit:
        return self.primitive

    def to_circuit_op(self) -> "CircuitOp":
        return self

    def to_instruction(self) -> Instruction:
        return self.primitive.to_instruction()

    # Warning - modifying immutable object!!
    def reduce(self) -> OperatorBase:
        if self.primitive.data is not None:
            # Need to do this from the end because we're deleting items!
            for i in reversed(range(len(self.primitive.data))):
                gate = self.primitive.data[i].operation
                # Check if Identity or empty instruction (need to check that type is exactly
                # Instruction because some gates have lazy gate.definition population)
                # pylint: disable=unidiomatic-typecheck
                if isinstance(gate, IGate) or (
                    type(gate) == Instruction and gate.definition.data == []
                ):
                    del self.primitive.data[i]
        return self

    def _expand_dim(self, num_qubits: int) -> "CircuitOp":
        return self.permute(list(range(num_qubits, num_qubits + self.num_qubits)))

    def permute(self, permutation: List[int]) -> "CircuitOp":
        r"""
        Permute the qubits of the circuit.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j of the circuit should be permuted to position permutation[j].

        Returns:
            A new CircuitOp containing the permuted circuit.
        """
        new_qc = QuantumCircuit(max(permutation) + 1).compose(self.primitive, qubits=permutation)
        return CircuitOp(new_qc, coeff=self.coeff)
